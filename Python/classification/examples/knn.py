import sys
sys.path.append('Python/classification')
from sklearn.neighbors import KNeighborsClassifier

import utils
import models
import numpy as np



if __name__ == "__main__":
    
    train, val, test = utils.load_data()
    knn = models.kNN()
    knn.fit(train[0], train[1])
    predicts = knn.predict(test[0], 1)
