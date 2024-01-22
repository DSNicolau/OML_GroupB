import datetime
import numpy as np
from scipy import signal
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


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


def normalise_time(data):
    for i in range(data.shape[1]):
        data[:, i] = (data[:, i] - np.min(data[:, i])) / (
            np.max(data[:, i]) - np.min(data[:, i])
        )
    return data


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
        
    else:
        raise NotImplementedError("Filter type not implemented.")
    return filtered_data


def remove_trend_seasonal(data, periodicity = int(60*24), model = "additive"): # 60 minutes * 24 hours
    datax = pd.DataFrame()
    data_return = pd.DataFrame()
    for i in data:
        if i in ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Week', 'Motion Detection']:
            data_return[i] = data[i]
            continue
        flipped_first_values = np.flip(data[i].head(periodicity).values)[int(periodicity/2):]
        flipped_last_values = np.flip(data[i].tail(periodicity).values)[:int(periodicity/2)]
        datax[i] = np.concatenate([flipped_first_values, data[i].values, flipped_last_values])
        result = seasonal_decompose(datax[i], model=model, period=periodicity)
        residual = result.resid.dropna().reset_index(drop=True)
        data_return[i] = residual
    return data_return