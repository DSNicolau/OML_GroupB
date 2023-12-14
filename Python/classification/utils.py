import pandas as pd
import numpy as np


def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B.xlsx", "Classification")
    data_pd.dropna(inplace=True)
    total_size = len(data_pd)
    train_size = int(total_size * 0.64)
    val_size = int(total_size * 0.16)
    train_data = data_pd.iloc[:train_size]
    val_data = data_pd.iloc[train_size : train_size + val_size]
    test_data = data_pd.iloc[train_size + val_size :]
    return train_data, val_data, test_data


def get_numpy_features(data):
    return data.iloc[:, :-1].to_numpy(), data.iloc[:, -1].to_numpy()


def moving_average(data, window_size=3):
    i = 0
    avgs = []

    while i < len(data) - window_size + 1:
        win = data[i : i + window_size]
        avg = np.mean(win, axis=0)
        avgs.append(avg)
        i += 1

    return avgs
