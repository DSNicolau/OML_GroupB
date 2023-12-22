import sys

sys.path.append("Python/classification/")

from utils import utils
import preprocessing
import evaluation
import plotly.express as px
import pandas as pd


if __name__ == "__main__":
    train, val, test = utils.load_data()
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    filtered_train_cheby2 = preprocessing.filter_day_periodicity(
        train_data,
        filter_type="cheby2",
        filter_order=5,
        rs=30,
        cutoff_frequency=1 / (2 * 24 * 60),
    )
    filtered_train_butter = preprocessing.filter_day_periodicity(
        train_data,
        filter_type="butter",
        filter_order=5,
        rs=30,
        cutoff_frequency=1 / (24 * 60),
    )
    filtered_train_cheby1 = preprocessing.filter_day_periodicity(
        train_data,
        filter_type="cheby1",
        filter_order=5,
        rs=30,
        cutoff_frequency=1 / (24 * 60),
    )
    df = pd.DataFrame(
        {
            "Unfiltered temperature": train_data[:, 0],
            "Filtered temperature - cheby2": filtered_train_cheby2[:, 0],
            "Filtered temperature - butter": filtered_train_butter[:, 0],
            "Filtered temperature - cheby1": filtered_train_cheby1[:, 0],
            "Unfiltered - Filtered cheby2": train_data[:, 0]
            - filtered_train_cheby2[:, 0],
            "Unfiltered - Filtered butter": train_data[:, 0]
            - filtered_train_butter[:, 0],
            "Unfiltered - Filtered cheby1": train_data[:, 0]
            - filtered_train_cheby1[:, 0],
            "label": train_label,
            "Time minutes": range(len(train_data)),
        }
    )
    fig = px.line(
        df,
        y=[
            "Unfiltered temperature",
            "Filtered temperature - cheby2",
            "Filtered temperature - butter",
            "Filtered temperature - cheby1",
            "Unfiltered - Filtered cheby2",
            "Unfiltered - Filtered butter",
            "Unfiltered - Filtered cheby1",
            "label"
        ],
        x="Time minutes",
    )
    fig.show()
