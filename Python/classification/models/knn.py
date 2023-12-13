import numpy as np
from .model import Model


class kNN(Model):
    def predict(self, x_test, n_neighbours=3, distance_type="euclidean"):
        if self.x_train.shape[0] < n_neighbours:
            raise ValueError(
                "The number of neighbours must be less than or equal to the number of training samples!"
            )
        if x_test.shape[1] != self.x_train.shape[1]:
            raise ValueError(
                "The number of features in x_test and x_train must be the same!"
            )
        dist = np.zeros((x_test.shape[0], n_neighbours))
        for i in range(x_test.shape[0]):
            dist[i] = self.y_train[
                np.argsort(self.distance(x_test[i], self.x_train, distance_type))[
                    :n_neighbours
                ]
            ]
        predicted_labels = np.round(np.mean(dist, axis=1))
        return predicted_labels
