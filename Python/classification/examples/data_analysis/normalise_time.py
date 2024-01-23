import sys

sys.path.append("Python/classification/")

import utils


if __name__ == "__main__":
    train, val, test = utils.load_data(normalise_time=True)
    train_data, train_label = utils.get_numpy_features(train, no_time=False)
    val_data, val_label = utils.get_numpy_features(val, no_time=False)
    test_data, test_label = utils.get_numpy_features(test, no_time=False)
    print(train)
    print(train_data.shape)




    