import numpy as np


def confusion_matrix(y_test, y_pred):
    tp = np.sum((y_test == 1) & (y_pred == 1))
    tn = np.sum((y_test == 0) & (y_pred == 0))
    fp = np.sum((y_test == 0) & (y_pred == 1))
    fn = np.sum((y_test == 1) & (y_pred == 0))
    cf_matrix = np.array([[tn, fp], [fn, tp]])
    return cf_matrix


def f1_score(cf_matrix):
    tp = cf_matrix[1][1]
    fp = cf_matrix[0][1]
    fn = cf_matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def precision_recall_f1_score(cf_matrix):
    tp = cf_matrix[1][1]
    fp = cf_matrix[0][1]
    fn = cf_matrix[1][0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score


def accuracy(cf_matrix):
    tp = cf_matrix[1][1]
    fp = cf_matrix[0][1]
    fn = cf_matrix[1][0]
    tn = cf_matrix[0][0]
    return (tp + tn) / (tp + tn + fp + fn)


def cohen_kappa(cf_matrix):
    tp = cf_matrix[1][1]
    fp = cf_matrix[0][1]
    fn = cf_matrix[1][0]
    tn = cf_matrix[0][0]
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        return 0
    return 2 * (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))


def evaluate(cf_matrix):
    acc = accuracy(cf_matrix)
    precision, recall, f1_score = precision_recall_f1_score(cf_matrix)
    cohen = cohen_kappa(cf_matrix)

    return acc, precision, recall, f1_score, cohen
