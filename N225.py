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

PATH = 'E:/AnacondaProjects/N225'
#PATH = 'C:/Users/s1887/AnacondaProjects/N225'
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
    volatility = pd.DataFrame(volatility)
    return volatility

#%%
data = volatility(N225)

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
num_classes = 5
target_column = 'NIKKEI225'

def make_dataset(data):
    
    inputs_data = []
    
    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    
    inputs_target = np.zeros(shape=(len(data)-time_length, num_classes))
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

train_data = data.loc['1950-05-17 00:00:00':'2017-12-29 00:00:00', :]

train_data_normalized, train_data_mean, train_train_std = zscore(train_data, train_data)

LSTM_inputs_train_data, LSTM_inputs_target_train_data = make_dataset(train_data_normalized)

#%%
# make Test data

test_data = data.loc['2018-01-04 00:00:00':'2019-02-01 00:00:00']

test_data_normalized, test_data_mean, test_data_std = zscore(train_data, test_data)

LSTM_inputs_test_data, LSTM_inputs_target_test_data = make_dataset(test_data_normalized)

#%%
fig, ax = plt.subplots(1, 1)
hist = train_data_normalized.hist(bins=300, ax=ax)
ax.vlines(-2.58, 0, 1000, colors='r', linewidth=0.8, label='-1%')
ax.vlines(2.58, 0, 1000, colors='r', linewidth=0.8, label='1%')
ax.legend()
plt.show()

#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
#from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

np.random.seed(123)

#%%
in_dim = LSTM_inputs_train_data.shape[2]
out_dim = num_classes
hidden_size = 64
batch_size = 128
epochs = 50

model = Sequential()
model.add(CuDNNLSTM(hidden_size, return_sequences=False,
               batch_input_shape=(None, time_length, in_dim)))
model.add(Dense(out_dim, activation='linear'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
        model_checkpoint = ModelCheckpoint(filepath='best_model_checkpint.h5', monitor='val_acc', save_best_only=True)
        LSTM_history = model.fit(LSTM_inputs_train_data, LSTM_inputs_target_train_data,
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 validation_split=0.3,
                                 shuffle=False,
                                 callbacks=[model_checkpoint])
        model.save_weights('LSTM_weights.h5')
        
        with open('LSTM_history.pickle', mode='wb') as f:
            pickle.dump(LSTM_history, f)
        break
    else:
        y_n = input("Wrong input caracter. Use saved weight? [y/n] : ")

#%%
loss = LSTM_history.history['loss']
val_loss = LSTM_history.history['val_loss']
acc = LSTM_history.history['acc']
val_acc = LSTM_history.history['val_acc']

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,9))
ax1.plot(range(len(loss)), loss, label='loss', color='blue')
ax1.plot(range(len(val_loss)), val_loss, label='val_loss', color='red')
ax1.set_xlabel('epochs', fontsize=12)
ax1.set_ylabel('loss', fontsize=12)
ax1.grid(which='major',color='gray',linestyle='--')
ax1.legend(fontsize=12)
ax2.plot(range(len(acc)), acc, label='acc', color='blue')
ax2.plot(range(len(val_acc)), val_acc, label='val_acc', color='red')
ax2.set_xlabel('epochs', fontsize=12)
ax2.set_ylabel('accuracy', fontsize=12)
ax2.grid(which='major',color='gray',linestyle='--')
ax2.legend(fontsize=12)
plt.savefig('model_loss_acc.png', dpi=300)
plt.show()

#%%
model.load_weights('best_model_checkpint.h5')
predicted_test_data = model.predict(LSTM_inputs_test_data)

print(predicted_test_data)

#%%
loss_and_metrics = model.evaluate(LSTM_inputs_test_data, LSTM_inputs_target_test_data)
print(loss_and_metrics)