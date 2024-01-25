import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set()

import sys
sys.path.append('Python/clustering/')
from models.k_means import K_Means

seed = 123
np.random.seed(seed)

# points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
#                   (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
#                   (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))

points = np.vstack(((np.random.randn(150, 3) * 0.75 + np.array([1, 0, 0])),
                  (np.random.randn(50, 3) * 0.25 + np.array([-0.5, 0.5, 0])),
                  (np.random.randn(50, 3) * 0.5 + np.array([-0.5, -0.5, 0]))))

num_trials = 5
k=3
model = K_Means(random_seed=seed)
k_means_centroids = model.fit(points, k, num_trials)
k_means_variances = model.trial_variances
print("k-means variances:\n", k_means_variances)
print("k-means centroids:\n", k_means_centroids)

from evaluation import plots
plots.plot3D_numpy(points, 
                   axis_labels=["Component 1", "Component 2", "Component 3"], 
                   title="k-means centroids",
                   point_size=5,
                   color=model.closest_centroid(),
                #    color=model.closest_centroid(),
                   color_map="viridis", 
                   show_colorbar=False)
