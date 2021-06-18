from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf

def get_crypto_data(symbol,**kwargs):
    interval = kwargs.get('interval','1d')
    start_date = kwargs.get('start_date', '2000-01-01-00-00')
    end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d-%H-%M'))
    seconds = 0
    if interval == '1d':
        seconds = 86400
    elif interval == '1h':
        seconds = 3600
    elif interval == '5m':
        seconds = 300
    elif interval == '1m':
        seconds = 60
    return HistoricalData(symbol,seconds,start_date,end_date).retrieve_data()

def get_stock_data(symbol,**kwargs):
    interval = kwargs.get('interval','1d')
    start_date = kwargs.get('start_date', '2000-01-01')
    end_date = kwargs.get('end_date', datetime.today().strftime('%Y-%m-%d'))
    period = kwargs.get('period', None)

    stock = yf.Ticker(symbol)
    if period != None:
        return stock.history(period=period,interval=interval)

    return stock.history(interval=interval,start=start_date,end=end_date)