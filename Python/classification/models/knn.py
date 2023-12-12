import numpy as np
from .model import Model

class kNN(Model):

    def predict(self, x_test, n_neighbours=3, distance_type="euclidean"):
        if self.x_train.shape[0] < n_neighbours:
            raise ValueError("The number of neighbours must be less than or equal to the number of training samples!")
        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError("The number of features in x_test and x_train must be the same!")
        
        dist = np.zeros((x_test.shape[0], self.x_train.shape[0]))
        for i in range(x_test.shape[0]):
            for j in range(self.x_train.shape[0]):
                dist[i, j] = self.distance(x_test[i], self.x_train[j], distance_type)
        indices = np.argsort(dist, axis=1)[:, :n_neighbours]
        nearest_labels = self.y_train[indices]
        predicted_labels = np.round(np.mean(nearest_labels, axis=1))
        return predicted_labels