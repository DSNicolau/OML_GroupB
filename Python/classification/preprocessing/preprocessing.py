import datetime
import numpy as np
import scipy
from scipy import signal


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


def filter_day_periodicity(
    data, filter_type="cheby2", filter_order=5, rs=30, cutoff_frequency=1 / (24 * 60)
):
    """ """

    # The cutoff frequency in Hz corresponds to one cycle per day
    filtered_data = np.zeros_like(data)
    if filter_type == "butter":
        b, a = signal.butter(
            N=filter_order, Wn=cutoff_frequency, btype="high", analog=False, output="ba"
        )
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
    elif filter_type == "cheby1":
        sos = signal.cheby1(
            filter_order, rs, cutoff_frequency, btype="high", output="sos"
        )
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.sosfilt(sos, data[:, i])
    elif filter_type == "cheby2":
        sos = signal.cheby2(
            filter_order, rs, cutoff_frequency, btype="high", output="sos"
        )
        for i in range(data.shape[1]):
            filtered_data[:, i] = signal.sosfilt(sos, data[:, i])
    elif filter_type=="ellip":
        
    else:
        raise NotImplementedError("Filter type not implemented.")
    return filtered_data
