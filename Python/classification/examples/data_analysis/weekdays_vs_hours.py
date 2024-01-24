import sys

sys.path.append("Python/classification/")

import utils
import preprocessing
import evaluation


if __name__ == "__main__":
    train, val, test = utils.load_data()
    train_data, train_label = utils.get_numpy_features(train, no_time=False)
    val_data, val_label = utils.get_numpy_features(val, no_time=False)
    test_data, test_label = utils.get_numpy_features(test, no_time=False)
    weeks = preprocessing.get_week_day(test_data)
    df_heatmap = preprocessing.number_events(
        x=weeks, y=test_data[:, 3], events=test_label, mean=True
    )
    weeks_name = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    evaluation.displayHeatMap(
        matrix=df_heatmap,
        x_label="Hours",
        y_label="Week day",
        y_tick_labels=weeks_name,
        title="Average hourly events per weekday - Test set",
    )
