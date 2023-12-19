import sys
sys.path.append('Python/classification/')
from utils import utils

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

def progressbar(it, prefix="", sufix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{} {}".format(prefix, u"█"*x, "."*(size-x), j, count, sufix), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

loaded_data = utils.load_data()
train_data, val_data, test_data = loaded_data
train_x, train_y = train_data
val_x, val_y = val_data
test_x, test_y = test_data

# Remove the time information from the data
train_x = train_x[:, 5:]
val_x = val_x[:, 5:]
test_x = test_x[:, 5:]

# Normalize the data
min_features = train_x.min(axis=0)
max_features = train_x.max(axis=0)

train_x = (train_x - min_features) / (max_features - min_features)
val_x = (val_x - min_features) / (max_features - min_features)
test_x = (test_x - min_features) / (max_features - min_features)

# Sequences creation
# def create_sequences(data_x, data_y, seq_length):
#     x = []
#     y = []
#     for i in range(data_x.shape[0] - seq_length):
#         x.append(data_x[i:i+seq_length])
#         y.append(data_y[i+seq_length - 1])
#     return np.array(x), np.array(y)

def create_sequences(data_x, data_y, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(data_x.shape[0]-lookback):
        # feature = torch.from_numpy(data_x[i:i+lookback])
        # # target = torch.from_numpy(dataset[i:i+lookback])
        # target = torch.from_numpy(data_y[i+1:i+lookback+1])
        feature = data_x[i:i+lookback]
        target = data_y[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    # return torch.tensor(X), torch.tensor(y)
    return torch.from_numpy(X).double(), torch.from_numpy(y)


# Define sequence length
seq_length = 10

# Create sequences for training set
X_train, y_train = create_sequences(train_x, train_y, seq_length)
# Create sequences for validation set
X_val, y_val = create_sequences(val_x, val_y, seq_length)
# Create sequences for test set
X_test, y_test = create_sequences(test_x, test_y, seq_length)

# Check shapes
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)


class MotionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=X_train.shape[2], hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

# Hardware Definition
device_indx = 0
avail_devices_count = torch.cuda.device_count()
actual_device_indx = device_indx if device_indx < avail_devices_count else avail_devices_count - 1
torch_device = "cuda:" + str(actual_device_indx)

device = torch.device(torch_device if torch.cuda.is_available() else "cpu")

model = MotionModel().double().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-6)
# loss_fn = nn.CrossEntropyLoss()
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=1028)
 
n_epochs = 10
for epoch in range(n_epochs):
    train_loss_epoch = 0
    # val_loss_epoch = 0
    train_accucuracy_epoch = 0
    
    model.train()
    # for X_batch, y_batch in loader:
    for X_batch, y_batch in progressbar(loader, "Train\tEpoch %d: " % (epoch + 1), "Batches", 40):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
                
        y_pred = model(X_batch)
        y_batch = y_batch.view(y_pred.shape)
        loss = loss_fn(y_pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
        last_y_pred = y_pred[:, -1, :]
        last_y_batch = y_batch[:, -1, :]
        last_y_pred[last_y_pred >= 0.5] = 1
        last_y_pred[last_y_pred < 0.5] = 0
        accuracy = (last_y_pred == last_y_batch).float().mean()
        train_accucuracy_epoch += accuracy
    # # Validation
    # if epoch % 100 != 0:
    #     continue
    model.eval()
    with torch.no_grad():
        X_val = X_val.to(device)
        y_val = y_val.to(device)
        y_pred = model(X_val)
        y_val = y_val.view(y_pred.shape)
        loss = loss_fn(y_pred, y_val)
        val_loss_epoch = loss.item()
        last_y_pred = y_pred[:, -1, :]
        last_y_val = y_val[:, -1, :]
        last_y_pred[last_y_pred >= 0.5] = 1
        last_y_pred[last_y_pred < 0.5] = 0
        val_accuracy = (last_y_pred == last_y_val).float().mean()
        
        
    train_loss_epoch /= len(loader)
    train_accucuracy_epoch /= len(loader)
    print("Epoch %d: Train Loss %.4f, Val Loss %.4f, Train Accuracy %.4f, Val Accuracy %.4f" % 
          (epoch, train_loss_epoch, val_loss_epoch, train_accucuracy_epoch, val_accuracy))
 
# with torch.no_grad():
#     # shift train predictions for plotting
#     train_plot = np.ones_like(timeseries) * np.nan
#     y_pred = model(X_train)
#     y_pred = y_pred[:, -1, :]
#     train_plot[lookback:train_size] = model(X_train)[:, -1, :]
#     # shift test predictions for plotting
#     test_plot = np.ones_like(timeseries) * np.nan
#     test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]

with torch.no_grad():
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    y_pred = model(X_test)
    print("y_pred shape: ", y_pred.shape)
    print("y_test shape: ", y_test.shape)
    y_test = y_test.unsqueeze(-1)
    loss = loss_fn(y_pred, y_test)
    test_loss_epoch = loss.item()
    last_y_pred = y_pred[:, -1, :]
    last_y_test = y_test[:, -1, :]
    last_y_pred[last_y_pred >= 0.5] = 1
    last_y_pred[last_y_pred < 0.5] = 0
    test_accuracy = (last_y_pred == last_y_test).float().mean()
    print("Test Loss %.4f, Test Accuracy %.4f" % (test_loss_epoch, test_accuracy))