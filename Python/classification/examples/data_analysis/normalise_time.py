import sys

sys.path.append("Python/classification/")

import utils
import preprocessing
import seaborn as sns
import evaluation


if __name__ == "__main__":
    train, val, test = utils.load_data()
    train_data, train_label = utils.get_numpy_features(train, no_time=False)
    val_data, val_label = utils.get_numpy_features(val, no_time=False)
    test_data, test_label = utils.get_numpy_features(test, no_time=False)
    weeks = preprocessing.get_week_day(test_data)
    