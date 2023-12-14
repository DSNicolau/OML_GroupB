from sklearn.feature_selection import mutual_info_classif

def mutual_information(x, y):
    return mutual_info_classif(x, y)