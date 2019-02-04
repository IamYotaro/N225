import pandas as pd
import numpy as np
import os
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pickle

'''
from google.colab import drive
drive.mount('/content/gdrive')
#PATH = '/content/gdrive/My Drive/AnacondaProjects/N225'
#os.chdir(PATH)
'''

#PATH = 'E:/AnacondaProjects/N225'
PATH = 'C:/Users/s1887/AnacondaProjects/N225'
#PATH = '/home/ky/AnacondaProjects/N225'
os.chdir(PATH)

#%%
N225 = web.DataReader("NIKKEI225","fred","1950/5/16")
N225.index = pd.to_datetime(N225.index)
N225.dropna(how='any', inplace=True)

N225.to_csv('N225.csv')

#%%
N225 = pd.read_csv('N225.csv')
N225.set_index('DATE', inplace=True)
N225.index = pd.to_datetime(N225.index)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(N225)
plt.show

#%%
volatility_column = 'NIKKEI225'

def volatility(data):
    volatility = np.zeros(len(data)-1)
    for i in range(len(data)-1):
        volatility[i] = 100 * np.log(data[volatility_column].iloc[i+1] / data[volatility_column].iloc[i])
    volatility = pd.Series(data=volatility, index=data.iloc[1:].index, name=volatility_column)
    return volatility

#%%
N225 = volatility(N225)
#%%
N225 = pd.DataFrame(N225)
#%%
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data, training_data_mean, training_data_std

def original_scale(predicted_data, training_data_mean, training_data_std):
    original_scale_predicted_data = training_data_std * predicted_data + training_data_mean
    return original_scale_predicted_data
    
confidence_interval_99 = [-2.58, 2.58]

#%%
time_length = 24
pred_category = 5
target_column = 'NIKKEI225'

def make_dataset(data):
    
    inputs_data = []
    
    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    
    inputs_target = np.zeros(shape=(len(data)-time_length, pred_category))
    for i in range(len(data)-time_length):
        if data[target_column][time_length + i] < confidence_interval_99[0]:
            inputs_target[i, 0] = 1
        elif confidence_interval_99[0] <= data[target_column][time_length + i] and data[target_column][time_length + i] < 0:
            inputs_target[i, 1] = 1
        elif data[target_column][time_length + i] == 0:
            inputs_target[i, 2] = 1
        elif 0 < data[target_column][time_length + i] and data[target_column][time_length + i] <= confidence_interval_99[1]:
            inputs_target[i, 3] = 1
        elif confidence_interval_99[0] < data[target_column][time_length + i]:
            inputs_target[i, 4] = 1

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target)

    return inputs_data_np, inputs_target_np

#%%
# make Training data

N225_train = N225.loc['1950-05-17 00:00:00':'2017-12-29 00:00:00', :]

N225_train_normalized, N225_train_mean, N225_train_std = zscore(N225_train, N225_train)

LSTM_inputs_data_train, LSTM_inputs_target_train = make_dataset(N225_train_normalized)

#%%
# make Test data

N225_test = N225.loc['2018-01-04 00:00:00':'2019-02-01 00:00:00']

N225_test_normalized, N225_test_mean, N225_test_std = zscore(N225_train, N225_test)

LSTM_inputs_data_test, LSTM_inputs_target_test = make_dataset(N225_test_normalized)

#%%
'''
N225_normalized, N225_mean, N225_std = zscore(N225, N225)
LSTM_inputs_data, LSTM_inputs_target = make_dataset(N225_normalized)

N225_test = N225[len(N225) - time_length:]
N225_test_normalized, N225_test_mean, N225_test_std = zscore(N225, N225_test)
LSTM_inputs_test_data = np.array(N225_test_normalized).reshape(1, time_length, len(N225_test_normalized.columns))
'''

#%%
from keras.models import Sequential
from keras.layers.core import Dense
#from keras.layers import CuDNNLSTM
from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

np.random.seed(123)

#%%
in_dim = LSTM_inputs_data_train.shape[2]
hidden_size = 64
out_dim = pred_category

model = Sequential()
model.add(LSTM(hidden_size, return_sequences=False,
               batch_input_shape=(None, time_length, in_dim)))
model.add(Dense(out_dim, activation='linear'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

#%%
y_n = input("Use saved weight? [y/n] : ")
while True:
    if y_n == 'y':
        model.load_weights('best_model_checkpint.h5')
        with open('LSTM_history.pickle', mode='rb') as f:
            LSTM_history = pickle.load(f)
        break
    elif y_n == 'n':
        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
        model_checkpoint = ModelCheckpoint(filepath='best_model_checkpint.h5', monitor='val_loss', save_best_only=True)
        LSTM_history = model.fit(LSTM_inputs_data_train, LSTM_inputs_target_train,
                                 batch_size=128,
                                 epochs=30,
                                 validation_split=0.1,
                                 shuffle=False,
                                 callbacks=[model_checkpoint])
        model.save_weights('LSTM_weights.h5')
        
        with open('LSTM_history.pickle', mode='wb') as f:
            pickle.dump(LSTM_history, f)
        break
    else:
        y_n = input("Wrong input caracter. Use saved weight? [y/n] : ")


#%%
base_line = mean_squared_error(LSTM_inputs_data_train[:,time_length-1,0], LSTM_inputs_target_train)
print('base line : %.5f'%base_line)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(LSTM_history.epoch, LSTM_history.history['loss'], label='training loss')
ax.plot(LSTM_history.epoch, LSTM_history.history['val_loss'], label='validation loss')
ax.hlines(base_line, 0, len(LSTM_history.epoch)-1, colors='r', linewidth=0.8, label='base line')
ax.annotate('base line: %.5f'%base_line, 
            xy=(0.72, 0.7),  xycoords='axes fraction',
            xytext=(0.72, 0.7), textcoords='axes fraction')
ax.set_title('model loss')
ax.set_ylabel('Mean Squared Error (MSE)',fontsize=12)
ax.set_xlabel('Epochs',fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("model_loss.png",dpi=300)
plt.show()

#%%
def MSE(data, pred_data):
    MSE = np.zeros(n_pred)
    for i in range(n_pred):
        MSE[i] = mean_squared_error(data[time_length+i:len(data)-n_pred+i+1]['Consumption'], pred_data[i])
    return MSE

#%%
# model evaluate
# mean absokute percentage error
    
def MAPE(data, pred_data):
    MAPE = np.zeros(n_pred)
    for i in range(n_pred):
        MAPE[i] = 100 * np.mean(np.abs((data[time_length+i:len(data)-n_pred+i+1]['Consumption'] - pred_data[i])/pred_data[i]))
    return MAPE

#%%
model.load_weights('best_model_checkpint.h5')
predicted_N225_test = model.predict(LSTM_inputs_data_test)
#%%
predicted_N225_test_original_scale = original_scale(predicted_N225_test, N225_train_mean['NIKKEI225'], N225_train_std['NIKKEI225'])

#%%
print(predicted_N225_test_original_scale)
