import sys

sys.path.append("Python/classification/")

import utils
import preprocessing
import evaluation

if __name__ == "__main__":
    train, val, test = utils.load_data()
    train_data, train_label = utils.get_numpy_features(train)
    val_data, val_label = utils.get_numpy_features(val)
    test_data, test_label = utils.get_numpy_features(test)
    train_fft =  preprocessing.get_fourrier_features(train_data[:, 5:])
    print('ah')