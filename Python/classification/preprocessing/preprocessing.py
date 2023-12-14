import datetime
import numpy as np
import pandas as pd


def get_week_day(data):
    weeks = [
        datetime.date(
            year=time[0].astype(int), month=time[1].astype(int), day=time[2].astype(int)
        ).isoweekday()
        for time in data
    ]
    return np.array(weeks)


def number_events(x, y, events, mean = False):
    matrix = np.zeros((len(np.unique(x)), len(np.unique(y))))
    if mean: 
        total = np.zeros((len(np.unique(x)), len(np.unique(y))))
    for i in range(len(x)):
        matrix[int(x[i])-1][int(y[i])] += events[i]
        if mean:
            total[int(x[i])-1][int(y[i])] += 1
    if mean:
        matrix = np.around(matrix/total, decimals=2)
    return matrix
