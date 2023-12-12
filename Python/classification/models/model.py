import numpy as np
class Model:
    def __init__(self):
        pass

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self):
        raise NotImplementedError

    def distance(self, x1, x2, p="euclidean"):
        distance = {"manhattan": 1, "euclidean": 2}
        if isinstance(p, str):
            try:
                p = distance[p]
            except KeyError:
                raise ValueError(
                    "Distance not implemented! Please try either manhattan or euclidean."
                )

        if not isinstance(p, int):
            raise ValueError("Distance type not supported! Please use integer.")

        # Minkowski distance
        return np.power(np.sum(np.power(np.abs(x1 - x2), p), axis=1), 1/p)

            
