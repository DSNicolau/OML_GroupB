import numpy as np

import sys
sys.path.append('Python/clustering/')
from models.model import Model


class K_Means(Model):
    
    def __init__(self, random_seed=None, distance_p_norm=2):
        """Initializes K-Means object

        Args:
            random_seed (int, optional): Random seed for reproducibility. If passed, the model will produce the same results across different runs. Defaults to None.
            distance_p_norm (int or str, optional): p-norm. Defaults to 2.

        Raises:
            ValueError: If random_seed is not an integer
        """
        super().__init__()
        if random_seed is not None:
            if not isinstance(random_seed, int):
                raise ValueError("Random seed must be an integer.")
        
        self.random_seed = random_seed
        self.distance_p_norm = distance_p_norm        
    
    def initialize_centroids(self):
        """returns k centroids from the initial points"""
        centroids = self.points.copy()
        np.random.shuffle(centroids)
        return centroids[:self.k]

    @staticmethod
    def distance(x1, x2, p : int or str ="euclidean", **kwargs):
        """ Calculates the p-norm distance between two row vectors

        Args:
            x1 (numpy.ndarray): input vector
            x2 (numpy.ndarray): input vector
            p (int or str, optional): p-norm. Defaults to "euclidean".

        Raises:
            ValueError: _p-norm must be an integer
            ValueError: _p-norm must be either manhattan or euclidean

        Returns:
            _type_: numpy.ndarray
        """
        if kwargs.get("axis") is not None:
            axis = kwargs.get("axis")
        else:
            axis = 1
        
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
        return np.power(np.sum(np.power(np.abs(x1 - x2), p), axis=axis), 1/p)

    def closest_centroid(self):
        """returns an array containing the index to the nearest centroid for each point"""
        distances = K_Means.distance(self.points, self.centroids[:, np.newaxis], p=self.distance_p_norm, axis=2)
        return np.argmin(distances, axis=0)

    def cluster_menbers(self):
        """returns an array containing the points for each cluster"""
        return np.array([self.points[self.closest==k] for k in range(self.centroids.shape[0])])

    def move_centroids(self):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([self.points[self.closest==k].mean(axis=0) for k in range(self.centroids.shape[0])])

    def clusters_variance(self):
        """returns the variance for each cluster"""
        return np.mean([K_Means.distance(self.points[self.closest==k], self.centroids[k], p=self.distance_p_norm).var(axis=0) for k in range(self.centroids.shape[0])])

    def fit(self, points, k, num_trials):
        """returns the best set of centroids found by running the algorithm num_trials times

        Args:
            points (numpy.ndarray): 2D array of data points with shape (n, m) where n is the number of data points and m is the number of features
            k (int): number of clusters
            num_trials (int): number of trials

        Returns:
            numpy.ndarray: 2D array of centroids with shape (k, m)
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        self.points = points.copy()
        self.k = k

        self.trial_variances = np.zeros(num_trials)
        self.trial_centroids = np.zeros((num_trials, k, self.points.shape[1]))


        for i in range(num_trials):
            print("Starting Trial: %d..." % (i))
            self.centroids = self.initialize_centroids()
            last_centroids = np.zeros_like(self.centroids) + np.inf

            while (last_centroids != self.centroids).any():
                last_centroids = self.centroids
                self.closest = self.closest_centroid()
                self.centroids = self.move_centroids()
                
            total_variance = self.clusters_variance()
            self.trial_variances[i] = total_variance
            self.trial_centroids[i] = self.centroids            
            
        self.best_trial_idx = np.argmin(self.trial_variances)
        self.centroids = self.trial_centroids[self.best_trial_idx]
        print("Best Trial: %d" % (self.best_trial_idx))
        print("Fitting Done!")
        return self.centroids