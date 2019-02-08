import pandas as pd
import numpy as np
import os
import pandas_datareader.data as pdr
import datetime
import fix_yahoo_finance as fix
fix.pdr_override()


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
start = datetime.date(2002, 6, 10)

def data_scraping_yahoo(file_name, target):
    data = pdr.DataReader(target,'yahoo', start)
    data.index = pd.to_datetime(data.index)
    data.dropna(how='any', inplace=True)
    data.rename(columns={'Adj Close': 'AdjClose'}, inplace=True)
    data.columns = [file_name+'_'+i for i in data.columns[:]]
    
    data.to_csv(os.path.join('DATA', '{name}.csv'.format(name=file_name)))

    return data

#%%
N225 = data_scraping_yahoo(file_name='N225', target='^N225')
DJIA = data_scraping_yahoo(file_name='DJIA', target='^DJI')

#%%
def volatility(data, data_name):
    volatility_column = [data_name+'_'+i for i in ['High', 'Low', 'Open', 'Close', 'AdjClose']]
    volatility = np.zeros(shape=(len(data)-1, len(volatility_column)))
    for i in range(len(volatility_column)):
        for j in range(len(data)-1):
            volatility[j, i] = 100 * np.log(data[volatility_column].iloc[j+1,i] / data[volatility_column].iloc[j,i])
    volatility = pd.DataFrame(data=volatility, index=data.iloc[1:].index, columns=volatility_column)
    return volatility

#%%
N225_volatility = volatility(data=N225, data_name='N225')
DJIA_volatility = volatility(data=DJIA, data_name='DJIA')

#%%
N225_volume = N225['N225_Volume'].iloc[1:]
DJIA_volume = DJIA['DJIA_Volume'].iloc[1:]
all_data = pd.concat([N225_volatility, N225_volume, DJIA_volatility, DJIA_volume], axis=1)
all_data.dropna(how='any', inplace=True)
all_data.to_csv(os.path.join('DATA', 'all_data.csv'))