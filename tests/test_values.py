from forecasting import torchLSTM as tl
from PythonDataProcessing import DataRetrieval as DR
import pandas
import yfinance
import numpy as np
import math
import pandas_ta as ta

def test_get_stock_data():
    ticker = 'AAPL'
    data_path = 'tests/dummy_data/'+ticker+'.csv'
    start_date = '2020-06-30'
    end_date = '2021-06-30'

    dummy_data = pandas.read_csv(data_path,index_col='Date')
    dummy_data =dummy_data.round(decimals=4)
    dummy_keys = dummy_data.keys().values

    data = DR.get_stock_data(ticker, start_date=start_date,end_date=end_date)
    data = data.round(decimals=4)
    keys = data.keys().values

    assert all(keys==dummy_keys)
    assert all(data['Open'].values == dummy_data['Open'].values)
    assert all(data['High'].values == dummy_data['High'].values)
    assert all(data['Low'].values == dummy_data['Low'].values)
    assert all(data['Close'].values == dummy_data['Close'].values)
    assert all(data['Volume'].values == dummy_data['Volume'].values)
    assert all(data['Dividends'].values == dummy_data['Dividends'].values)
    assert all(data['Stock Splits'].values == dummy_data['Stock Splits'].values)

def test_sma():
    # Parameters
    ticker = 'AAPL'
    data_path = 'tests/dummy_data/'+ticker+'_MA.csv'
    scale = 30
    key = 'Close'

    # Loading test data
    dummy_data = pandas.read_csv(data_path,index_col='Date')

    # Getting SMA
    data = DR.add_SMA(dummy_data,key=key,scale=scale)
    sma = data['SMA'].values

    # Checking that
    assert all(np.isnan(sma[0:scale-1]))
    assert not all(np.isnan(sma[0:scale]))
    assert len(data) == len(dummy_data)

    # Manual Check that start and end values are correct
    dummy_close = data[key].values
    assert round(sum(dummy_close[0:scale])/scale,4) == round(sma[scale-1],4)
    assert round(sum(dummy_close[-scale:])/scale,4) == round(sma[-1],4)

    #Checking with csv data
    dummy_sma = dummy_data['SMA-30']
    assert all(np.round(dummy_sma.values[scale-1:],4) == np.round(sma[scale-1:],4))


def test_relnorm_fit():
    # Parameters
    ticker = 'AAPL'
    data_path = 'tests/dummy_data/' + ticker + '.csv'
    dummy_data_path = 'tests/dummy_data/' + ticker + '_Rel_Norm.csv'
    keys = ['High','Low','Close','Volume']

    # Initialize Scaler
    data_scaler = DR.CustomScaler()
    norm_range = 200
    data_scaler.mx_range = norm_range

    # Loading test data
    data = pandas.read_csv(data_path, index_col='Date')
    dummy_data = pandas.read_csv(dummy_data_path, index_col='Date')

    # Fit the scaler
    data_scaler.fit(data[keys])

    # Get maxes and mins                    #
    closeMax = data_scaler.mx_dict['Close']
    closeMin= data_scaler.mn_dict['Close']

    # Get maxes and min dummy data
    dummy_closeMax = dummy_data['CloseMax'].values[norm_range-1:]
    dummy_closeMin = dummy_data['CloseMin'].values[norm_range-1:]

    assert all(np.round((closeMax),5) == np.round(dummy_closeMax,5))
    assert all(np.round((closeMin),5) == np.round(dummy_closeMin,5))


def test_relnorm_transform():
    # Parameters
    ticker = 'AAPL'
    data_path = 'tests/dummy_data/' + ticker + '.csv'
    dummy_data_path = 'tests/dummy_data/' + ticker + '_Rel_Norm.csv'
    keys = ['High', 'Low', 'Close', 'Volume']
    dummy_keys = ['HighNorm','LowNorm','CloseNorm','VolumeNorm']

    # Initialize Scaler
    data_scaler = DR.CustomScaler()
    norm_range = 200
    data_scaler.mx_range = norm_range

    # Loading test data
    data = pandas.read_csv(data_path, index_col='Date')
    dummy_data = pandas.read_csv(dummy_data_path, index_col='Date')

    # Fit the scaler
    data_scaler.set_removal(norm_range)
    data_scaler.fit(data[keys])
    data=data.iloc[norm_range-1:,:]

    # Get transformed data
    norm = data_scaler.transform(data[keys])

    # Load dummy tranformed data
    dummy_norm = dummy_data[dummy_keys].iloc[norm_range-1:,:].values

    assert all(np.round(dummy_norm[:,0],4) == np.round(norm[:,0],4))
    assert all(np.round(dummy_norm[:,1],4) == np.round(norm[:,1],4))
    assert all(np.round(dummy_norm[:,2],4) == np.round(norm[:,2],4))
    assert all(np.round(dummy_norm[:,3],4) == np.round(norm[:,3],4))


'''
Todo:
- Finish testing all fits.
'''