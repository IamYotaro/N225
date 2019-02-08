import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt

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
all_data.drop(drop_list, axis=1, inplace=True)

#%%
fig, ax = plt.subplots(1,1)
ax.plot(N225['N225_Close'])
plt.show

#%%
x = np.array(N225.iloc[i: i + 11 , 3])                
(ca, cd) = pywt.dwt(x, "haar")                
cat = pywt.threshold(ca, np.std(ca), mode="soft")                
cdt = pywt.threshold(cd, np.std(cd), mode="soft")                
tx = pywt.idwt(cat, cdt, "haar")

#%%
fig, ax = plt.subplots(1,1, figsize=(16,9))
ax.plot(N225['N225_Close'].values, label='Actual')
ax.plot(tx, label='WaceletTransform')
plt.legend(fontsize=12)
plt.show