import pandas as pd
import numpy as np
import os

'''
from google.colab import drive
drive.mount('/content/gdrive')
#PATH = '/content/gdrive/My Drive/AnacondaProjects/N225'
#os.chdir(PATH)
'''

PATH = 'E:/AnacondaProjects/N225'
os.chdir(PATH)

#%%
N225 = pd.read_csv('^N225.csv', sep=',')
def data_arrange(data, date_column, drop_column):
    data.drop(drop_column, axis=1, inplace=True)
    pd.to_datetime(data[date_column])
    data.set_index(date_column, inplace=True)
    data.replace('null', np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.astype(np.float64)
    return data

N225 = data_arrange(data=N225, date_column='Date', drop_column=['Volume', 'Adj Close'])
N225.head()

#%%
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data, training_data_mean, training_data_std

def original_scale(predicted_data, training_data_mean, training_data_std):
    original_scale_predicted_data = training_data_std * predicted_data + training_data_mean
    return original_scale_predicted_data

#%%
time_length = 24
def make_dataset(data, target_column):
    
    inputs_data = []
    
    for i in range(len(data)-time_length):
        temp_set = data[i:(i+time_length)].copy()
        inputs_data.append(temp_set)
    inputs_target = data[target_column][time_length:]

    inputs_data_np = [np.array(inputs_data) for inputs_data in inputs_data]
    inputs_data_np = np.array(inputs_data_np)
    
    inputs_target_np = np.array(inputs_target).reshape(len(inputs_data), 1)

    return inputs_data_np, inputs_target_np

#%%
N225_normalized, N225_mean, N225_std = zscore(N225, N225)
LSTM_inputs_data, LSTM_inputs_target = make_dataset(N225_normalized, target_column='Close')

N225_test = N225[len(N225) - time_length:]
N225_test_normalized, N225_test_mean, N225_test_std = zscore(N225, N225_test)
LSTM_inputs_test_data = np.array(N225_test_normalized).reshape(1, time_length, len(N225_test_normalized.columns))


#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.metrics import mean_squared_error

np.random.seed(123)

#%%
in_dim = LSTM_inputs_data.shape[2]
hidden_size = 858
out_dim = 1

model = Sequential()
model.add(CuDNNLSTM(hidden_size, return_sequences=False,
               batch_input_shape=(None, time_length, in_dim)))
model.add(Dense(out_dim, activation='linear'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
plot_model(model, to_file='model.png', show_shapes=True)

#%%
y_n = input("Use saved weight? [y/n] : ")
while True:
    if y_n == 'y':
        model.load_weights('best_model_checkpint.h5')
        break
    elif y_n == 'n':
        early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
        model_checkpoint = ModelCheckpoint(filepath='best_model_checkpint.h5', monitor='val_loss', save_best_only=True)
        LSTM_history = model.fit(LSTM_inputs_data, LSTM_inputs_target,
                                 batch_size=128,
                                 epochs=10,
                                 validation_split=0.1,
                                 shuffle=False,
                                 callbacks=[model_checkpoint])
        model.save_weights('LSTM_weights.h5')
        break
    else:
        y_n = input("Wrong input caracter. Use saved weight? [y/n] : ")


#%%
base_line = mean_squared_error(LSTM_inputs_data[:,time_length-1,1], LSTM_inputs_target)
print('base line : %.5f'%base_line)

#%%
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

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
model.load_weights('best_model_checkpint.h5')
predicted_N225_test = model.predict(LSTM_inputs_test_data)
predicted_N225_test_original_scale = original_scale(predicted_N225_test, N225_mean['Close'], N225_std['Close'])


#%%
print(predicted_N225_test_original_scale)
