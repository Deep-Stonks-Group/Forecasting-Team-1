from forecasting import torchLSTM as tl
from PythonDataProcessing import DataRetrieval as DR

def test_file_saving_and_loading():
    ticker = 'AAPL'
    epochs = 0
    training_set_coeff = 0.9
    period = '2y'
    interval = '1h'

    predictor = tl.PredictionEngine(ticker,epochs=epochs,training_set_coeff=training_set_coeff,period=period,interval=interval)
    predictor.train_ticker()
    predictor.save_model()
    predictor = tl.PredictionEngine(ticker,is_loading=True)

    assert epochs == predictor.epochs
    assert training_set_coeff == predictor.data_handler.training_set_coeff
    assert period == predictor.data_handler.period
    assert interval == predictor.data_handler.interval

def test_default_training():
    ticker = 'AAPL'

    predictor = tl.PredictionEngine(ticker, epochs=200)  # Creates model
    predictor.train_ticker()  # Trains model

def test_relative_Norm():
    ticker = 'CCL'

    predictor = tl.PredictionEngine(ticker, epochs=200, normalizer_type='Relative')  # Creates model
    predictor.train_ticker()  # Trains model

def test_learning_rate():
    ticker = 'FB'

    predictor = tl.PredictionEngine(ticker, epochs=200, learning_rate=0.05)  # Creates model
    predictor.train_ticker()  # Trains model

def test_input_features():
    ticker = 'JPM'

    predictor = tl.PredictionEngine(ticker, epochs=200, input_features=['Open', 'Low', 'Volume'])  # Creates model
    predictor.train_ticker()  # Trains model

def test_label_features():
    ticker = 'SNAP'

    predictor = tl.PredictionEngine(ticker, epochs=200, input_features=['Open'])  # Creates model
    predictor.train_ticker()  # Trains model

def test_indicator_key():
    ticker = 'TSLA'

    predictor = tl.PredictionEngine(ticker, epochs=200, indicator_key='Open')  # Creates model
    predictor.train_ticker()  # Trains model

def test_seq_length():
    ticker = 'ETH-USD'

    predictor = tl.PredictionEngine(ticker, epochs=200, seq_length=20)  # Creates model
    predictor.train_ticker()  # Trains model

def test_interval():
    ticker = 'CMCSA'

    predictor = tl.PredictionEngine(ticker, epochs=200, interval='1m',period='5d')  # Creates model
    predictor.train_ticker()  # Trains model

def test_period():
    ticker = 'JPM'

    predictor = tl.PredictionEngine(ticker, epochs=200, period='5y')  # Creates model
    predictor.train_ticker()  # Trains model

def test_start_date():
    ticker = 'DIS'

    predictor = tl.PredictionEngine(ticker, epochs=200, start_date='2000-01-01')  # Creates model
    predictor.train_ticker()  # Trains model

def test_end_date():
    ticker = 'HD'

    predictor = tl.PredictionEngine(ticker, epochs=200, end_date='2021-01-01')  # Creates model
    predictor.train_ticker()  # Trains model

def test_training_set_coeff():
    ticker = 'XOM'

    predictor = tl.PredictionEngine(ticker, epochs=200, training_set_coeff=0.8)  # Creates model
    predictor.train_ticker() # Trains model

def test_use_coinbase():
    ticker = 'AAVE-USD'

    predictor = tl.PredictionEngine(ticker, epochs=200, use_coinbase=True)  # Creates model
    predictor.train_ticker() # Trains model

def test_train_stocks():
    top_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
    for stock in top_stocks:
        predictor = tl.PredictionEngine(stock, epochs=100, training_set_coeff=1)  # Creates model
        predictor.train_ticker()  # Trains model
        pos,pred_now,pred_prev = predictor.predict_now(period='2y')

def test_train_cryptos():
    top_cryptos = DR.get_all_currencies()
    for crypto in top_cryptos:
        ticker = crypto + '-USD'
        predictor = tl.PredictionEngine(ticker, epochs=100, training_set_coeff=1)  # Creates model
        predictor.train_ticker()  # Trains model
        pos,pred_now,pred_prev = predictor.predict_now(period='2y')

def test_full_train_hourly():
    ticker = 'SQFT'
    predictor = tl.PredictionEngine(ticker,training_set_coeff=0.9,period='2y',interval='1h',normalizer_type='Relative') #Creates model
    predictor.train_ticker() # Trains model
    predictor.eval_ticker() # Plots and prints results
    predictor.save_model()
    pos,pred_now,pred_prev = predictor.predict_now(period='3mo')

def test_full_train_daily():
    ticker = 'MCK'
    predictor = tl.PredictionEngine(ticker,training_set_coeff=0.9,period='max',normalizer_type='Relative') #Creates model
    predictor.train_ticker()
    predictor.eval_ticker()
    pos,pred_now,pred_prev = predictor.predict_now(period='2y')
