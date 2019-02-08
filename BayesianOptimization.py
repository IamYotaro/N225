import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

'''
from google.colab import drive
drive.mount('/content/gdrive')
#PATH = '/content/gdrive/My Drive/AnacondaProjects/N225'
#os.chdir(PATH)
'''

PATH = 'E:/AnacondaProjects/N225'
#PATH = 'C:/Users/s1887/AnacondaProjects/N225'
#PATH = '/home/ky/AnacondaProjects/N225'
os.chdir(PATH)

#%%
def read_data(file_name):
    data = pd.read_csv(os.path.join('DATA', file_name))
    data.set_index('Date', inplace=True)
    data.index = pd.to_datetime(data.index)
    return data

#%%
all_data = read_data(file_name='all_data.csv')

drop_list = ['N225_Volume', 'DJIA_Volume']
all_data.drop(drop_list, axis=1, inplace=True)

#%%
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data
    
confidence_interval_99 = [-2.58, 2.58]

#%%
num_classes = 4
target_column = 'N225_Close'

def make_dataset(data, time_length):
    
    inputs_data = []
    
    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    
    inputs_target = np.zeros(shape=(len(data)-time_length, num_classes))
    for i in range(len(data)-time_length):
        if data[target_column][time_length + i] <= confidence_interval_99[0]:
            inputs_target[i, 0] = 1
        elif confidence_interval_99[0] < data[target_column][time_length + i] and data[target_column][time_length + i] < 0:
            inputs_target[i, 1] = 1
        elif 0 <= data[target_column][time_length + i] and data[target_column][time_length + i] < confidence_interval_99[1]:
            inputs_target[i, 2] = 1
        elif confidence_interval_99[0] <= data[target_column][time_length + i]:
            inputs_target[i, 3] = 1

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target)

    return inputs_data_np, inputs_target_np

#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
#from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal
from bayes_opt_custom import BayesianOptimization

np.random.seed(123)

#%%
def model_by_BayesianOptimization(time_length,
                                  hidden_size_1,                                                           
                                  batch_size):
    
    time_length = int(time_length)
    
    train_data = all_data.loc[:'2017-10-31 00:00:00']
    train_data_normalized = zscore(train_data, train_data)
    LSTM_inputs_train_data, LSTM_inputs_target_train_data = make_dataset(train_data_normalized, time_length)
    
    validation_data = all_data.loc['2017-11-01 00:00:00':'2018-07-31 00:00:00']
    validation_data_normalized = zscore(train_data, validation_data)
    LSTM_inputs_validation_data, LSTM_inputs_target_validation_data = make_dataset(validation_data_normalized, time_length)

    in_dim = LSTM_inputs_train_data.shape[2]
    out_dim = num_classes
    hidden_size = int(hidden_size_1)
    batch_size = int(batch_size)
    epochs = 10
    
    model = Sequential()
    model.add(CuDNNLSTM(hidden_size, return_sequences=False,
                        batch_input_shape=(None, time_length, in_dim),
                        kernel_initializer = glorot_uniform(seed=123),
                        recurrent_initializer = orthogonal(gain=1.0, seed=123)))
    model.add(Dense(out_dim, activation='softmax',
                    kernel_initializer = glorot_uniform(seed=123)))

    Adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    model.compile(loss='kullback_leibler_divergence', optimizer=Adamax, metrics=['categorical_accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='categorical_accuracy', mode='auto', patience=10)
    model_checkpoint = ModelCheckpoint(filepath=os.path.join('BayesianOptimization', 'best_model_checkpint.h5'), monitor='val_loss', save_best_only=True, mode='auto')
    model.fit(LSTM_inputs_train_data,
              LSTM_inputs_target_train_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(LSTM_inputs_validation_data, LSTM_inputs_target_validation_data),
              shuffle=False,
              callbacks=[early_stopping, model_checkpoint])
    model.save_weights(os.path.join('BayesianOptimization', 'LSTM_weights.h5'))
    
    model.load_weights(os.path.join('BayesianOptimization', 'best_model_checkpint.h5'))

    loss_and_metrics = model.evaluate(LSTM_inputs_validation_data, LSTM_inputs_target_validation_data, verbose = 1)

    return loss_and_metrics[1]

#%%
# BayesianOptimization
init_points = 2
n_iter = 3

pbounds = {'time_length': (5, 60),
           'hidden_size_1': (32, 1024),
           'batch_size': (32, 1024)}

optimizer = BayesianOptimization(f=model_by_BayesianOptimization, pbounds=pbounds)

optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ei')

#%%
all_BO = optimizer.res['all']
max_BO = optimizer.res['max']

print(max_BO)

#%%
with open(os.path.join('BayesianOptimization', 'all_BO.pickle'), 'wb') as f:
    pickle.dump(all_BO, f)
    
all_BO_values = np.zeros(n_iter)
all_BO_time_length = np.zeros(n_iter)
all_BO_hidden_size_1 = np.zeros(n_iter)
all_BO_batch_size = np.zeros(n_iter)
for i in range(n_iter):
    all_BO_values[i] = all_BO['values'][i]
    all_BO_time_length[i] = all_BO['params'][i]['time_length']
    all_BO_hidden_size_1[i] = all_BO['params'][i]['hidden_size_1']
    all_BO_batch_size[i] = all_BO['params'][i]['batch_size']
    
all_BO_np = np.vstack((all_BO_values, all_BO_time_length, all_BO_hidden_size_1, all_BO_batch_size)).T
all_BO_pd = pd.DataFrame(data=all_BO_np, columns=['categorical_accuracy', 'time_length', 'hidden_size_1', 'batch_size'])