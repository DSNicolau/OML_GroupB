import sys

sys.path.append("Python/classification/")

from utils import utils
import models
import evaluation
import time


if __name__ == "__main__":
    train, _, test = utils.load_data()
    knn = models.kNN()
    train_data, train_label = utils.get_numpy_features(train)
    knn.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test)
    print('gonna start')
    start = time.time()
    predicts = knn.predict(test_data, n_neighbours=1, chunk_size=1000)
    print(time.time() - start)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(cf_matrix)
