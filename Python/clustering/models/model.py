import numpy as np
class Model:
    def __init__(self):
        pass

    def fit(self, data, **kwargs):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    @staticmethod
    def distance(self, x1, x2, p="euclidean", **kwargs):
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

            
