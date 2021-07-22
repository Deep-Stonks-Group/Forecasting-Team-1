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
        :param ticker: Sets the ticker you want to train on. *Required*
        :param kwargs: Adjust the input for training.
            input_features: What data do you want to use for training?
            label_features: What do you want to use as a label?
            seq_length: How many days/hours/etc. should be used for each input into LSTM.
            interval: Should input be in terms of minutes/hours/days/weeks/months/years
            period: How long should the input data span over?
            training_set_coeff: What percent of data should be used for training?
        '''
        self.ticker = ticker
        self.input_features = kwargs.get('input_features', ['High','Low','Close','Volume','EMA'])
        self.label_features = kwargs.get('label_features', ['SMA'])
        self.seq_length = kwargs.get('seq_length', 10)
        self.interval = kwargs.get('interval', '1d')
        self.period = kwargs.get('period', '2y')
        self.start_date = kwargs.get('start_date', None)
        self.end_date = kwargs.get('end_date', None)
        self.training_set_coeff = kwargs.get('training_set_coeff', 0.8)
        self.epochs = kwargs.get('epochs', 2000)
        self.learning_rate = kwargs.get('learning_rate', 0.01)
        loading_model = kwargs.get('loading_model', False)
        self.scaler = DR.Scaler(self.input_features,self.label_features)
        if not loading_model:
            self.scaler.data_scaler = MinMaxScaler()
            self.scaler.label_scaler = MinMaxScaler()
            self.x,self.y = self.get_data()

    def create_model(self, input_size=5, hidden_size=6, num_layers=1, num_classes=1):
        '''
        Initializes LSTM to be used for training.
        '''
        self.lstm = LSTM(num_classes, input_size, hidden_size, num_layers, self.seq_length)

    def save_model(self):
        '''
        Saves the model under simple_lstm/models/ so that it can be used int future.
        '''
        model = {k:v for k,v in self.__dict__.items()}
        self.lstm.trained_tickers.sort()
        name = ''.join(self.lstm.trained_tickers)
        with open('simple_lstm/models/' + name + '.p', 'wb') as outfile:
            pickle.dump(model, outfile)

    def load_model(self, name):
        '''
        Loads model from simple_lstm/models/
        :param name: Name of the model you want to load.
        '''
        if name:
            try:
                with open('simple_lstm/models/' + name + '.p', 'rb') as infile:
                    model = pickle.load(infile)
            except Exception as e:
                print('could not load model {}'.format(name))
                raise
            for k,v in model.items():
                self.__dict__[k] = v

    def get_data(self):
        '''
        This function uses DataRtrieval to load the data, normalize it and format it to be used by the LSTM.
        :return x: The input data for the model after it is loaded, split and ready to be used.
        :return y: The labels for the model after it is loaded and adjusted.
        '''
        data = DR.get_stock_data(self.ticker, interval=self.interval,
                                            period=self.period, start_date=self.start_date, end_date=self.end_date)
        data = DR.add_features(self.input_features, self.label_features, data)

        self.train_size = int(len(data) * self.training_set_coeff)
        self.scaler.set_train_size(self.train_size)
        self.scaler.fit_scalers(data)
        scaled_data, scaled_labels = self.scaler.normalize_stock_data(data)
        x, y = DR.sliding_windows(scaled_data, scaled_labels, self.seq_length)

        return x,y

    def train_ticker(self):
        '''
        Loads the data, and gets the training data. Then passes the model and training data to train_model.
        :return: Returns if model is already trained.
        '''
        if not hasattr(self, 'lstm'):
            self.create_model()
        self.lstm.train()
        if self.ticker in self.lstm.trained_tickers:
            print(f'Already trained model on {self.ticker}')
            return

        train_x = torch.Tensor(self.x[0:self.train_size])
        train_y = torch.Tensor(self.y[0:self.train_size])
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
        test_x = torch.Tensor(self.x[self.train_size:])
        test_y = torch.Tensor(self.y[self.train_size:])

        all_predict = self.lstm(dataX)
        data_predict = all_predict.data.numpy()
        dataY_plot = dataY.data.numpy()

        data_predict = self.scaler.label_scaler.inverse_transform(data_predict)
        dataY_plot = self.scaler.label_scaler.inverse_transform(dataY_plot.reshape(dataY_plot.shape[0],1))

        plt.axvline(x=self.train_size, c='r', linestyle='--')
        plt.plot(dataY_plot)
        plt.plot(data_predict)
        plt.suptitle('Time-Series Prediction')
        plt.show()
        MET.print_metrics(test_x,test_y,self.lstm,self.scaler.label_scaler)

    def predict(self, input_sequence):
        '''

        :param input_sequence:
        :return:
        '''
        scaled_input_sequence = self.scaler.data_scaler.transform(input_sequence)
        scaled_input_sequence = torch.tensor(scaled_input_sequence).float()
        scaled_input_sequence = torch.reshape(scaled_input_sequence,[1,10,5])
        output = self.lstm(scaled_input_sequence)
        prediction = self.scaler.label_scaler.inverse_transform(output.data.numpy())[0][0]
        return prediction

    def predict_now(self):
        data = DR.get_stock_data(self.ticker, interval=self.interval, period=self.period)
        data = DR.add_features(self.input_features, self.label_features, data)
        data = data.tail(self.seq_length)
        return predictor.predict(data[['High', 'Low', 'Close', 'Volume', 'EMA']])





predictor = PredictionEngine('AAPL')
# predictor.load_model('AAPL')
# print(predictor.predict_now())

predictor.create_model()
predictor.train_ticker()
predictor.eval_ticker()
predictor.save_model()


"""
top_stocks = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'CMCSA', 'JPM', 'HD', 'DIS', 'XOM']
for stock in top_stocks:
    predictor = PredictionEngine()
    predictor.train_ticker(stock)
    predictor.save_model()
"""


