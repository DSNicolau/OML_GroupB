import sys

sys.path.append("Python/classification/")

import utils
import preprocessing
import seaborn as sns
import evaluation

if __name__ == "__main__":
    train, val, test = utils.load_data()
    train_data, train_label = utils.get_numpy_features(train)
    val_data, val_label = utils.get_numpy_features(val)
    test_data, test_label = utils.get_numpy_features(test)
    print('ah')
    # x = preprocessing.mutual_information(train[0][:, 5:], train[1])
    weeks = preprocessing.get_week_day(train_data)
    df_heatmap = preprocessing.number_events(
        x=weeks, y=train_data[:, 3], events=train_label, mean=True
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
