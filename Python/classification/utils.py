import pandas as pd
import numpy as np


def load_data():
    data_pd = pd.read_excel("data/Datasets_Group_B.xlsx", "Classification")
    data_pd.ffill(inplace=True)
    # data_pd.fillna( method ='ffill', inplace = True)
    data_np = data_pd.to_numpy()
    total_size = data_np.shape[0]
    train_size = int(total_size * 0.8)
    val_size = int(total_size * 0.16)
    train_data = data_np[0:train_size, :]
    val_data = data_np[train_size : train_size + val_size, :]
    test_data = data_np[train_size + val_size :, :]
    return (
        (train_data[:, :-1], train_data[:, -1]),
        (val_data[:, :-1], val_data[:, -1]),
        (test_data[:, :-1], test_data[:, -1]),
    )


def moving_average(data, window_size=3):
    i = 0
    avgs = []

    while i < len(data) - window_size + 1:
        win = data[i : i + window_size]
        avg = np.mean(win, axis=0)
        avgs.append(avg)
        i += 1

    return avgs

