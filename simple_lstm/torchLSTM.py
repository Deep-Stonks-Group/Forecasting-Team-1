import numpy as np 
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size) #reshape output from 1,train_len,hidden to train_len,hidden
        out = self.fc(h_out)
        return out

def prepare_data(data, labels, seq_length):
    x = []
    y = []
    for i in range(data.shape[0]-seq_length-1):
        _x = data[i: (i+seq_length)]
        _y = labels[i+seq_length]
        x.append(_x)
        y.append(_y)
    return x, y

#df = web.DataReader('SPX',data_source='yahoo',start='2012-01-01',end='2021-05-13',retry_count=5)
#df.to_csv('SPX.csv')
data_scaler = MinMaxScaler()
labl_scaler = MinMaxScaler()
data_source = pd.read_csv('DIA.csv')
# Date High Low Open Close Volume AdjClose
scaled_data = data_scaler.fit_transform(data_source[['High','Low','Volume']]) 
scaled_lbls = labl_scaler.fit_transform(data_source[['Close']])
#normed = data_source[data_source.columns[1:]].apply(lambda x: x/x.max())

seq_length = 10
x, y = prepare_data(scaled_data, scaled_lbls, seq_length)
dataX = torch.Tensor(np.array(x))
dataY = torch.Tensor(np.array(y))
train_size = int(len(y) * 0.75)
train_x = torch.Tensor(x[0:train_size])
train_y = torch.Tensor(y[0:train_size])
test_x = torch.Tensor(x[train_size:])
test_y = torch.Tensor(y[train_size:])

input_size = 3
hidden_size = 6
num_layers = 1
num_classes = 1
epochs = 2000
learning_rate = 0.01

lstm = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

"""TRAIN"""
lstm.train()
for epoch in range(epochs):
    outputs = lstm(train_x)
    optimizer.zero_grad()
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

"""TEST"""
lstm.eval()
train_predict = lstm(dataX)
data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = labl_scaler.inverse_transform(data_predict)
dataY_plot = labl_scaler.inverse_transform(dataY_plot.reshape(dataY_plot.shape[0],1))

plt.axvline(x=train_size, c='r', linestyle='--')
plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()


print('done')
