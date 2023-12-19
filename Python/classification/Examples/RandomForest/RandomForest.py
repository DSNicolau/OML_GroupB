import sys
sys.path.append('Python/')
import os 

from utils import utils

import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
        y.append([data_y[i+seq_length - 1]])
        # y.append(data_y[i+1:i+seq_length+1])
    return np.array(x), np.array(y)

def balance_dataset(x, y, rate_pos):
    print("Before balancing:")
    print("Shape of x: ", x.shape)
    print("Shape of y: ", y.shape)
    pos = x[y == 1]
    neg = x[y == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    t_n = neg.shape[0]
    t_p = pos.shape[0]
    t_p_new = (t_n * rate_pos) // (1 - rate_pos)
    reminer_p = t_p_new - t_p
    times_t_p = int(reminer_p // t_p)
    reminer_p = int(reminer_p - (t_p*times_t_p))
    idx = list(range(t_p))*times_t_p + list(range(reminer_p))
    new_x = np.concatenate((x, pos[idx]))
    new_y = np.concatenate((y, np.ones(int(t_p_new - t_p))))
    # new_y = np.concatenate((y, np.ones(int(t_p_new - t_p)).reshape(-1, 1)))
    print("After balancing:")
    print("Shape of x: ", new_x.shape)
    print("Shape of y: ", new_y.shape)
    pos = new_x[new_y == 1]
    neg = new_x[new_y == 0]
    print("Positive samples: ", pos.shape[0])
    print("Negative samples: ", neg.shape[0])
    return new_x, new_y

# seq_length = 1

# # Create sequences for training set
# X_train, y_train = create_sequences(train_x, train_y, seq_length)
# # Create sequences for validation set
# X_val, y_val = create_sequences(val_x, val_y, seq_length)
# # Create sequences for test set
# X_test, y_test = create_sequences(test_x, test_y, seq_length)

X_train, y_train = train_x, train_y
X_val, y_val = val_x, val_y
X_test, y_test = test_x, test_y

# Balance dataset
print("Balancing dataset...")
print("Training set:")
X_train, y_train = balance_dataset(X_train, y_train, rate_pos=0.5)
print("Validation set:")
X_val, y_val = balance_dataset(X_val, y_val, rate_pos=0.5)
print("Test set:")
X_test, y_test = balance_dataset(X_test, y_test, rate_pos=0.5)

clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)

clf.fit(X_train, y_train)


# Calculate Accuracy
from sklearn.metrics import accuracy_score

# Train Accuracy
train_y_pred = clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_y_pred)
print("Train Accuracy:", train_accuracy)

y_pred = clf.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)