import sys
sys.path.append('../../')
from utils import utils
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = utils.load_data()
train_data, val_data, test_data = data
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
def create_sequences(data_x, data_y, seq_length):
    x = []
    y = []
    for i in range(data_x.shape[0] - seq_length):
        x.append(data_x[i:i+seq_length])
        y.append(data_y[i+seq_length])
    return np.array(x), np.array(y)

# Define sequence length
seq_length = 50

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

# Building the LSTM Model

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dense(units=1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                    epochs=100, batch_size=1, verbose=2)

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Evaluate the model
score = model.evaluate(X_test, y_test)

