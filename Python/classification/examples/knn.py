import sys
sys.path.append('Python/classification')

import utils
import models
import numpy as np



if __name__ == "__main__":
    
    x = np.array([[1,2,3],[4,5,6],[7,8,9], [10,11,12], [1,5,7]])
    x_label = np.array([1,1,1,0,0])
    # y = np.array([[4,3,5],[1,5,7]])
    y = np.array([[1,5,7]])
    knn = models.kNN()
    knn.fit(x, x_label)
    print(knn.predict(y, 1))
