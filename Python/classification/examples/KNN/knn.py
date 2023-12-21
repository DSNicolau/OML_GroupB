import sys

sys.path.append("Python/classification/")

from utils import utils
import models
import evaluation


if __name__ == "__main__":
    train, _, test = utils.load_data(normalize=True)
    knn = models.kNN()
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    knn.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)
    print('starting')
    predicts = knn.predict(test_data, n_neighbours=1)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(cf_matrix)
