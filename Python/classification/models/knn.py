import numpy as np
from .distances import distance

class kNN:
    def __init__(self, n_neighbours=3):
        self.n_neighbours = n_neighbours

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test, distance_type="euclidean"):
        if self.x_train.shape[0] < self.n_neighbours:
            raise ValueError("The number of neighbours must be less than or equal to the number of training samples!")
        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError("The number of features in x_test and x_train must be the same!")
        
        dist = np.zeros((x_test.shape[0], self.x_train.shape[0]))
        for i in range(x_test.shape[0]):
            for j in range(self.x_train.shape[0]):
                dist[i, j] = distance(x_test[i], self.x_train[j], distance_type)
        indices = np.argsort(dist, axis=1)[:, :self.n_neighbours]
        nearest_labels = self.y_train[indices]
        predicted_labels = np.round(np.mean(nearest_labels, axis=1))
        return predicted_labels