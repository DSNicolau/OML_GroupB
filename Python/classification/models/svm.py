import sys

sys.path.append("Python/classification/")

from utils import utils
import models
import evaluation


if __name__ == "__main__":
    train, val, test = utils.load_data(normalize=True)
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)
    val_data, val_label = utils.get_numpy_features(val, no_time=True)