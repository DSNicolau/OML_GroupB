import sys

sys.path.append("Python/classification/")

import utils
import models
import evaluation


if __name__ == "__main__":
    train, _, test = utils.load_data()
    knn = models.kNN()
    train_data, train_label = utils.get_numpy_features(train)
    knn.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test)
    predicts = knn.predict(test_data, n_neighbours=1)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(cf_matrix)
