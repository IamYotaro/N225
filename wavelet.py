import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from statsmodels.robust import mad

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

train_data = read_data(file_name='train_data.csv')

#%%
fig, ax = plt.subplots(1,1)
ax.plot(N225['N225_Close'])
plt.show

#%%
def waveletSmooth(data, wavelet='haar', level=1, title=None):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec(data, wavelet, mode='per')
    # calculate a threshold
    sigma = mad(coeff[-level])
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt(2*np.log(len(data)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:])
    # reconstruct the signal using the thresholded coefficients
    reconstruct_data = pywt.waverec(coeff, wavelet, mode='per')
    
    return reconstruct_data

#%%
N225_wavelet = waveletSmooth(N225['N225_Close'])

fig, ax = plt.subplots(1,1, figsize=(16,9))
ax.plot(N225['N225_Close'].values, label='Actual')
ax.plot(N225_wavelet, label='WaceletTransform')
ax.grid(which='major',color='gray',linestyle='--')
plt.legend(fontsize=12)
plt.savefig(os.path.join('wavelet', 'N225_Close_wavelet_transform.png'), dpi=300)
plt.show

#%%
train_data_wavelet = []
for i in range(len(train_data.columns)):
    temp_wavelet = np.array(waveletSmooth(train_data.iloc[:,i]))
    
    if i == 0:
        train_data_wavelet = temp_wavelet
    else:
        train_data_wavelet = np.vstack([train_data_wavelet, temp_wavelet])
train_data_wavelet = train_data_wavelet.T

#%%
fig, ax = plt.subplots(1,1, figsize=(16,9))
ax.plot(all_data['N225_Close'].values, label='Actual')
ax.plot(train_data_wavelet[:,3], label='WaceletTransform')
ax.grid(which='major',color='gray',linestyle='--')
plt.legend(fontsize=12)
plt.savefig(os.path.join('wavelet', 'all_data_N225_Close_wavelet_transform.png'), dpi=300)
plt.show

#%%
def log_diff(data):
    data_log_diff = np.diff(np.log(data), n=1, axis=0)*100
    return data_log_diff

train_data_wavelet_log_diff = log_diff(train_data_wavelet)
np.savetxt(os.path.join('DATA','train_data_wavelet_log_diff.csv'), train_data_wavelet_log_diff, delimiter=',')

print(np.any(np.isinf(train_data_wavelet_log_diff) == 'True'))


#%%
x = np.array(N225['N225_Close'])
cA, cD = pywt.dwt(x, 'haar')
cA_threshold = pywt.threshold(cA, np.std(cA), mode='hard')
cD_threshold = pywt.threshold(cD, np.std(cD), mode='hard')
# threshold is standard deviation. less than standard deviation is changed zero.
# Specificaly -sigma < value < -sigma is zero. 
x_wavelet = pywt.idwt(cA_threshold, cD_threshold, 'haar')

#%%
fig, ax = plt.subplots(1,1, figsize=(16,9))
ax.plot(all_data['N225_Close'].values, label='Actual')
ax.plot(x_wavelet, label='WaceletTransform')
ax.grid(which='major',color='gray',linestyle='--')
plt.legend(fontsize=12)
plt.savefig(os.path.join('wavelet', 'all_data_N225_Close_wavelet_transform.png'), dpi=300)
plt.show


#%%

a = [1,2,3,4,5,6,7,8]
(cA,cD) = pywt.dwt(a,'db1')
print(cA)
print(cD)

b = pywt.idwt(cA,cD,'db1')
print(b)

Coeffs = pywt.wavedec(a,'db1')
print(Coeffs)

depth = pywt.dwt_max_level(len(a),len(pywt.Wavelet('db1')))
print(depth)