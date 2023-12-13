import numpy as np


def confusion_matrix(y_test, y_pred):
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    cf_matrix = np.array([[tn, fp], [fn, tp]])
    return cf_matrix
