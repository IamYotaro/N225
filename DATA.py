import pandas as pd
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

all_data = pd.concat([N225, DJIA], axis=1)
all_data.dropna(how='any', inplace=True)
all_data.to_csv(os.path.join('DATA', 'all_data.csv'))

#%%
drop_list = ['N225_Volume', 'DJIA_Volume']
all_data.drop(drop_list, axis=1, inplace=True)

train_data = all_data.loc[:'2017-10-31 00:00:00']
validation_data = all_data.loc['2017-11-01 00:00:00':'2018-07-31 00:00:00']
test_data = all_data.loc['2018-08-01 00:00:00':]

train_data.to_csv(os.path.join('DATA', 'train_data.csv'))
validation_data.to_csv(os.path.join('DATA', 'validation_data.csv'))
test_data.to_csv(os.path.join('DATA', 'test_data.csv'))