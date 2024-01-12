import sys

sys.path.append("Python/classification/")

from utils import utils
from sklearn import svm
import evaluation

if __name__ == "__main__":
    train, _, test = utils.load_data(normalize=True)
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    clf = svm.SVC(kernel="linear")
    clf.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)
    predicts = clf.predict(test_data)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(cf_matrix)
