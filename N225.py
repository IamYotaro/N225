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
N225 = read_data(file_name='N225.csv')
DJIA = read_data(file_name='DJIA.csv')

all_data = read_data(file_name='all_data.csv')

drop_list = ['N225_Volume', 'DJIA_Volume']
all_data.drop(drop_list, axis=1, inplace=False)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(N225['N225_Close'])
plt.show

#%%
def zscore(training_data, data):
    training_data_mean = np.mean(training_data, axis=0)
    training_data_std = np.std(training_data, axis=0)
    normalized_data = (data - training_data_mean) / training_data_std
    return normalized_data
    
confidence_interval_99 = [-2.58, 2.58]

#%%
time_length = 59
num_classes = 4
target_column = 'N225_Close'

def make_dataset(data):
    
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
# make Training data

train_data = all_data.loc[:'2017-10-31 00:00:00']

train_data_normalized = zscore(train_data, train_data)

LSTM_inputs_train_data, LSTM_inputs_target_train_data = make_dataset(train_data_normalized)

#%%
# make Validation data
validation_data = all_data.loc['2017-11-01 00:00:00':'2018-07-31 00:00:00']

validation_data_normalized = zscore(train_data, validation_data)

LSTM_inputs_validation_data, LSTM_inputs_target_validation_data = make_dataset(validation_data_normalized)

#%%
# make Test data

test_data = all_data.loc['2018-08-01 00:00:00':]

test_data_normalized = zscore(train_data, test_data)

LSTM_inputs_test_data, LSTM_inputs_target_test_data = make_dataset(test_data_normalized)

#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
hist1 = train_data_normalized['N225_Close'].hist(bins=300, ax=ax1, label='train_data')
ax1.vlines(-2.58, 0, 120, colors='r', linewidth=0.8, label='-0.5%')
ax1.vlines(2.58, 0, 120, colors='r', linewidth=0.8, label='0.5%')
ax1.legend()
hist2 = test_data_normalized['N225_Close'].hist(bins=200, ax=ax2, label='test_data')
ax2.vlines(-2.58, 0, 16, colors='r', linewidth=0.8, label='-0.5%')
ax2.vlines(2.58, 0, 16, colors='r', linewidth=0.8, label='0.5%')
ax2.legend()
plt.savefig(os.path.join('model', 'hist_train_data.png'), dpi=300)
plt.show()

#%%
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import CuDNNLSTM
#from keras.layers.recurrent import LSTM
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.initializers import orthogonal

np.random.seed(123)

#%%
in_dim = LSTM_inputs_train_data.shape[2]
out_dim = num_classes
hidden_size = 125
batch_size = 302
epochs = 100

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
plot_model(model, to_file=os.path.join('model', 'model.png'), show_shapes=True)

early_stopping = EarlyStopping(monitor='categorical_accuracy', mode='auto', patience=10)
model_checkpoint = ModelCheckpoint(filepath=os.path.join('model', 'best_model_checkpint.h5'), monitor='val_loss', save_best_only=True, mode='auto')
LSTM_history = model.fit(LSTM_inputs_train_data, LSTM_inputs_target_train_data,
                         batch_size=batch_size,
                         epochs=epochs,
                         validation_data=(LSTM_inputs_validation_data, LSTM_inputs_target_validation_data),
                         shuffle=False,
                         callbacks=[early_stopping, model_checkpoint])
model.save_weights(os.path.join('model', 'LSTM_weights.h5'))
        
with open(os.path.join('model', 'LSTM_history.pickle'), mode='wb') as f:
    pickle.dump(LSTM_history, f)

loss = LSTM_history.history['loss']
val_loss = LSTM_history.history['val_loss']
acc = LSTM_history.history['categorical_accuracy']
val_acc = LSTM_history.history['val_categorical_accuracy']

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(16,9))
ax1.plot(range(len(loss)), loss, label='loss', color='blue', linestyle='-.')
ax1.plot(range(len(val_loss)), val_loss, label='val_loss', color='red')
ax1.set_xlabel('epochs', fontsize=12)
ax1.set_ylabel('loss', fontsize=12)
ax1.grid(which='major',color='gray',linestyle='--')
ax1.legend(fontsize=12)
ax2.plot(range(len(acc)), acc, label='acc', color='blue', linestyle='-.')
ax2.plot(range(len(val_acc)), val_acc, label='val_acc', color='red')
ax2.set_xlabel('epochs', fontsize=12)
ax2.set_ylabel('accuracy', fontsize=12)
ax2.grid(which='major',color='gray',linestyle='--')
ax2.legend(fontsize=12)
plt.savefig(os.path.join('model', 'model_loss_acc.png'), dpi=300)
plt.show()

#%%
model.load_weights(os.path.join('model', 'best_model_checkpint.h5'))
predicted_test_data = model.predict(LSTM_inputs_test_data)

loss_and_metrics = model.evaluate(LSTM_inputs_test_data, LSTM_inputs_target_test_data, verbose = 1)

judgement = []
if np.argmax(predicted_test_data[-1,:]) == np.argmax(LSTM_inputs_target_test_data[-1,:]):
    judgement = 'OK'
else:
    judgement = 'NO'
    
    
print('loss:', loss_and_metrics[0])
print('categorical_accuracy:', loss_and_metrics[1])
print('predicted_last_test_data:', predicted_test_data[-1])
print('predicted_last_test_data_category: {category} {judgement}'.format(category=np.argmax(predicted_test_data[-1,:]), judgement=judgement))

#%%
tomorrow_pred_data = all_data.iloc[len(all_data)-time_length:]
tomorrow_pred_data_normalized = zscore(train_data, tomorrow_pred_data)

LSTM_inputs_tomorrow_pred_data = np.resize(tomorrow_pred_data.values, [1, time_length, in_dim])

predicted_tomorrow_pred_data = model.predict(LSTM_inputs_tomorrow_pred_data)

print('next_day_prediction:', predicted_tomorrow_pred_data[-1])
print('next_day_prediction_category:', np.argmax(predicted_tomorrow_pred_data))