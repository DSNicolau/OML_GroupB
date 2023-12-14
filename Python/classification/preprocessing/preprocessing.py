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


def number_events(x, x_name, y, events):
    # Create a pandas dataframe that counts how many events (motion detections) happen per event x and y
    data_dict={key: np.zeros(len(np.unique(y))) for key in x_name}
    translation = {np.unique(x)[i]: x_name[i] for i in range(len(x_name))}

    for i in range(len(x)):
        data_dict[translation[x[i]]][int(np.unique(y)[i])] += events[i]
    return data_dict
