import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PythonDataProcessing import DataRetrieval as DR
from PythonDataProcessing import Metrics as MET
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True).float()

        self.fc = nn.Linear(hidden_size, num_classes)
        self.trained_tickers = []
    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size) #reshape output from 1,train_len,hidden to train_len,hidden
        out = self.fc(h_out)
        return out

def train_model(lstm, train_x, train_y, epochs=2000, learning_rate=.01):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    lstm.train()
    for epoch in range(epochs):
        outputs = lstm(train_x)
        optimizer.zero_grad()
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

class PredictionEngine():

    def __init__(self,ticker: str,**kwargs):
        '''
        Initialization of the data handler.

          Input:
              *ticker* (Required) - String ticker that you want to train on. \n
              **kwargs**
                  *epochs* (optional) - Number of epochs used for training.
                  *learning_rate* (optional) - Learning rate used for training.
                  *is_loading* (optional) - Indicates if you are loading or creating a model.

                  *input_features* (Optional) - Array of features to be used for input. \n
                  *label_features* (Optional) - Array of feature to be used for label. \n
                  *indicator_key* (Optional) - String indicating which column in data you want to use for indicator. \n
                  *seq_length* (Optional) - Integer length of sequence. \n
                  *interval* (Optional) - String interval of prices: 1d, 1h, 5m or 1m. \n
                  *period* (Optional) - String specify a length of data to be retrieved: 1d, 1y, 1m, 2y \n
                  *start_date* (Optional) - The earliest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
                  *end_date* (Optional) - The latest historic data you want in form YYYY-MM-DD: 2000-01-01 \n
                  *training_set_coeff* (Optional) - Decimal percent of dataset used for training. \n
                  *is_crpyto* (Optional) - Boolean which indicates if ticker is a crypto. \n
                  *normalizer_type (Optional) - String which indicates the type of normalizer to use: 'Relative' or 'MinMax'*
        '''
        self.ticker = ticker

        # Training Parameters
        self.epochs = kwargs.get('epochs', 2000)
        self.learning_rate = kwargs.get('learning_rate', 0.01)

        # Initializing the data and scalers
        is_loading = kwargs.get('is_loading', False)
        self.data_handler = DR.LSTM_DATA_HANDLE(ticker,**kwargs)
        if not is_loading:
            self.x,self.y = self.data_handler.retrieve_data()
            self.create_model(input_size=len(self.data_handler.input_features))
        else:
            self.load_model(ticker)

    def create_model(self, input_size=5, hidden_size=6, num_layers=1, num_classes=1):
        '''
        Initializes LSTM to be used for training.

        Inputs:
            *input_size* (optional) - Integer of features in input data. \n
            *hidden_size* (optional) - Integer of hidden nodes. \n
            *num_layers* (optional) - Integer of number of layers. \n
            *num_classes* (optional) - Integer of labels associated with single input. \n
        '''
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, self.data_handler.seq_length)

    def save_model(self):
        '''
        Saves the model under simple_lstm/models/ so that it can be used in the future.
        '''
        model = {k:v for k,v in self.__dict__.items()}
        self.lstm.trained_tickers.sort()
        name = ''.join(self.lstm.trained_tickers)
        with open('forecasting/models/' + name + '.p', 'wb') as outfile:
            pickle.dump(model, outfile)

    def load_model(self, name):
        '''
        Loads model from simple_lstm/models/ and all the other arguments.
        Inputs:
            *name* (required): Name of the model you want to load.
        '''
        if name:
            try:
                with open('forecasting/models/' + name + '.p', 'rb') as infile:
                    model = pickle.load(infile)
            except Exception as e:
                print('could not load model {}'.format(name))
                raise
            for k,v in model.items():
                self.__dict__[k] = v


    def train_ticker(self):
        '''
        Loads the data, and gets the training data. Then passes the model and training data to train_model.
        '''
        if not hasattr(self, 'lstm'):
            self.create_model()
        self.lstm.train()
        if self.ticker in self.lstm.trained_tickers:
            print(f'Already trained model on {self.ticker}')
            return

        train_x = torch.Tensor(self.x[0:self.data_handler.train_size])
        train_y = torch.Tensor(self.y[0:self.data_handler.train_size])
        train_model(self.lstm, train_x, train_y,epochs=self.epochs,learning_rate=self.learning_rate)
        self.lstm.trained_tickers.append(self.ticker)

    def eval_ticker(self):
        '''
        Loads the data and splits it into the test set. Passes the test set into the trained model.
        Plot/print the results.
        '''
        self.lstm.eval()

        dataX = torch.Tensor(np.array(self.x))
        dataY = torch.Tensor(np.array(self.y))
        test_x = torch.Tensor(self.x[self.data_handler.train_size:])
        test_y = torch.Tensor(self.y[self.data_handler.train_size:])

        all_predict = self.lstm(dataX)
        data_predict = all_predict.data.numpy()
        dataY_plot = dataY.data.numpy()

        data_predict = self.data_handler.label_scaler.inverse_transform(data_predict,10)
        dataY_plot = self.data_handler.label_scaler.inverse_transform(dataY_plot.reshape(dataY_plot.shape[0],1),10)

        plt.axvline(x=self.data_handler.train_size, c='r', linestyle='--')
        plt.plot(dataY_plot)
        plt.plot(data_predict)
        plt.suptitle('Time-Series Prediction')
        plt.show()
        MET.print_metrics(test_x,test_y,self.lstm,self.data_handler)

    def predict(self, input_sequence):
        '''
            Generates a prediction for a given input sequence.
            Inputs:
                *input_sequence* (required) - Data frame of historic data with correct input_features.

            Outputs:
                *prediction* - The predicted future price.

            Example:
                predict(data[['High','Low','Close','Volume','EMA']])

        '''
        scaled_input_sequence = self.data_handler.data_scaler.transform(input_sequence)
        scaled_input_sequence = torch.tensor(scaled_input_sequence).float()
        scaled_input_sequence = torch.reshape(scaled_input_sequence,[1,10,5])
        output = self.lstm(scaled_input_sequence)
        prediction = self.data_handler.label_scaler.inverse_transform(output.data.numpy(),10)[0][0]
        return prediction

    def predict_now(self):
        '''
            Generates a prediction for the next minute/hour/day.
            Outputs:
                *prediction* - The predicted future price.
        '''
        data = DR.get_stock_data(self.ticker, interval=self.data_handler.interval, period=self.data_handler.period)
        data = DR.add_features(self.data_handler.input_features, self.data_handler.label_features, data)
        data = data.tail(self.data_handler.seq_length)
        return predictor.predict(data[self.data_handler.input_features])



# Create and train a model with all default values
ticker = 'CCL'
predictor = PredictionEngine(ticker,epochs=200,normalizer_type='Relative') #Creates model
predictor.train_ticker() # Trains model
predictor.eval_ticker() # Plots and prints results
predictor.save_model() # Stores model


'''
# Create and train an hourly crypto model.
ticker = 'ETH-USD'
predictor = PredictionEngine(ticker, is_crypto=True,interval='1h',start_date='2021-04-01-00-00') #Creates moddel
predictor.train_ticker() # Trains model
predictor.eval_ticker() # Plots and prints results
predictor.save_model() # Stores model
'''

''' # Load a model and see the next days prediction
ticker = 'CCL'
predictor = PredictionEngine(ticker, is_loading=True) #Creates model
print(predictor.predict_now()) # Print prediction for next day.
'''

""" # Loop through top 10 stocks and train a model.
top_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
for stock in top_stocks:
    predictor = PredictionEngine(stock,training_set_coeff=1) #Creates model
    predictor.train_ticker() # Trains model
    predictor.save_model() # Stores model
"""

""" # Loop through top 10 stocks and print a prediction.
top_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
for stock in top_stocks:
    predictor = PredictionEngine(stock, is_loading=True)
    print(predictor.predict_now())
"""

