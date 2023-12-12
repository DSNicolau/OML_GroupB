import sys
sys.path.append('Python/classification')

import utils
import models
import numpy as np



if __name__ == "__main__":
    
    train, val, test = utils.load_data()
    knn = models.kNN()
    knn.fit(train[0], train[1])
    print(knn.predict(train[0], 1))
