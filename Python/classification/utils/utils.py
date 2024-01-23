import pandas as pd
import numpy as np
from preprocessing import remove_trend_seasonal, get_week_day, filter_day_periodicity


def load_data(normalize=False, rmv_trend_seasonal=False, normalise_time=False, filt = None):
    data_pd = pd.read_excel("data/Datasets_Group_B_v2.xlsx", "Classification")
    data_pd.dropna(inplace=True)
    total_size = len(data_pd)
    train_size = int(total_size * 0.64)
    val_size = int(total_size * 0.16)
    train_data = data_pd.iloc[:train_size].reset_index(drop=True)
    val_data = data_pd.iloc[train_size : train_size + val_size].reset_index(drop=True)
    test_data = data_pd.iloc[train_size + val_size :].reset_index(drop=True)
    if filt:
        filtered_train = filter_day_periodicity(
            train_data.iloc[:, 5:-1].to_numpy(),
            filter_type=filt,
            filter_order=5,
            rs=30,
            cutoff_frequency=1 / (24 * 60),
        )
        train_data.iloc[:, 5:-1] = filtered_train

        filtered_test = filter_day_periodicity(
            test_data.iloc[:, 5:-1].to_numpy(),
            filter_type=filt,
            filter_order=5,
            rs=30,
            cutoff_frequency=1 / (24 * 60),
        )
        test_data.iloc[:, 5:-1] = filtered_test

        filtered_val = filter_day_periodicity(
            val_data.iloc[:, 5:-1].to_numpy(),
            filter_type=filt,
            filter_order=5,
            rs=30,
            cutoff_frequency=1 / (24 * 60),
        )
        val_data.iloc[:, 5:-1] = filtered_val
    if rmv_trend_seasonal:
        train_data = remove_trend_seasonal(train_data)
        val_data = remove_trend_seasonal(val_data)
        test_data = remove_trend_seasonal(test_data)
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
    if normalise_time:
        train_np, _ = get_numpy_features(train_data, no_time=False)
        val_np, _ = get_numpy_features(val_data, no_time=False)
        test_np, _ = get_numpy_features(test_data, no_time=False)

        train_data['Weeks'] = get_week_day(train_np)
        val_data['Weeks'] = get_week_day(val_np)
        test_data['Weeks'] = get_week_day(test_np)
        train_data.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        val_data.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        test_data.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        norms = [[1,7],[0,23],[0,59]]
        to_normalize = ['Weeks','Hour','Minute']
        for i in range(3):
            train_data[to_normalize[i]] = min_max_normalization_pandas(train_data[to_normalize[i]], min_values=norms[i][0], max_values=norms[i][1])
            val_data[to_normalize[i]] = min_max_normalization_pandas(val_data[to_normalize[i]], min_values=norms[i][0], max_values=norms[i][1])
            test_data[to_normalize[i]] = min_max_normalization_pandas(test_data[to_normalize[i]], min_values=norms[i][0], max_values=norms[i][1])
    


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
