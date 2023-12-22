import datetime
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def get_week_day(data):
    weeks = [
        datetime.date(
            year=time[0].astype(int), month=time[1].astype(int), day=time[2].astype(int)
        ).isoweekday()
        for time in data
    ]
    return np.array(weeks)


def number_events(x, y, events, mean=False):
    matrix = np.zeros((len(np.unique(x)), len(np.unique(y))))
    if mean:
        total = np.zeros((len(np.unique(x)), len(np.unique(y))))
    for i in range(len(x)):
        matrix[int(x[i]) - 1][int(y[i])] += events[i]
        if mean:
            total[int(x[i]) - 1][int(y[i])] += 1
    if mean:
        matrix = np.around(matrix / total, decimals=2)
    return matrix


def filter_day_periodicity(data):
    # The cutoff frequency in Hz corresponds to one cycle per day
    cutoff_frequency = 1 / (24 * 60)
    filter_order = 4

    b, a = butter(
        N=filter_order, Wn=cutoff_frequency, btype="high", analog=False, output="ba"
    )
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered_data[:, i] = filtfilt(b, a, data[:, i])
    return filtered_data
