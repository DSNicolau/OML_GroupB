import pandas as pd
import numpy as np


def load_data(normalize=False):
    data_pd = pd.read_excel("data/Datasets_Group_B_v2.xlsx", "Classification")
    data_pd.dropna(inplace=True)
    total_size = len(data_pd)
    train_size = int(total_size * 0.64)
    val_size = int(total_size * 0.16)
    train_data = data_pd.iloc[:train_size]
    val_data = data_pd.iloc[train_size : train_size + val_size]
    test_data = data_pd.iloc[train_size + val_size :]
    if normalize:
        train_data.iloc[:, 5:-1], min_values, max_values = min_max_normalization_pandas(
            train_data.iloc[:, 5:-1]
        )
        val_data.iloc[:, 5:-1] = min_max_normalization_pandas(
            val_data.iloc[:, 5:-1], min_values, max_values
        )
        test_data.iloc[:, 5:-1] = min_max_normalization_pandas(
            test_data.iloc[:, 5:-1], min_values, max_values
        )

    return train_data, val_data, test_data


def min_max_normalization_pandas(dataframe, min_values=None, max_values=None):
    if min_values is None or max_values is None:
        min_values = dataframe.min(axis=0)
        max_values = dataframe.max(axis=0)
        normalized_dataframe = (dataframe - min_values) / (max_values - min_values)

        return normalized_dataframe, min_values, max_values
    
    else:
        normalized_dataframe = (dataframe - min_values) / (max_values - min_values)

        return normalized_dataframe


def get_numpy_features(data, no_time=True):
    if no_time:
        data, labels = data.iloc[:, 5:-1].to_numpy(), data.iloc[:, -1].to_numpy()
    else:
        data, labels = data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy()

    return data, labels


def min_max_normalization(data):
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    normalized_data = (data - min_values) / (max_values - min_values)

    return normalized_data


def moving_average(data, window_size=3):
    i = 0
    avgs = []

    while i < len(data) - window_size + 1:
        win = data[i : i + window_size]
        avg = np.mean(win, axis=0)
        avgs.append(avg)
        i += 1

    return avgs


def load_data_v2():
    data_pd = pd.read_excel("data/Datasets_Group_B_v2.xlsx", "Classification")
    # data_pd.fillna(method='ffill', inplace=True)
    data_pd.dropna(inplace=True)
    data_np = data_pd.to_numpy()
    total_size = data_np.shape[0]
    num_positive = (data_np[:, -1] == 1).sum()
    per_positive = num_positive / total_size
    print("Total size: ", total_size)
    print("Number of positive: ", num_positive)
    print("Percentage of positive: ", per_positive)
    train_size = int(total_size * 0.64)
    val_size = int(total_size * 0.16)
    train_data = data_np[0:train_size, :]
    val_data = data_np[train_size : train_size + val_size, :]
    test_data = data_np[train_size + val_size :, :]

    return (
        (train_data[:, :-1], train_data[:, -1]),
        (val_data[:, :-1], val_data[:, -1]),
        (test_data[:, :-1], test_data[:, -1]),
    )
