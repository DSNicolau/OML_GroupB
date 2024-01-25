import sys
sys.path.append('Python/')
from clustering.models.k_means import K_Means
from clustering.utils.utils import load_data, min_max_nomalization

from sklearn.cluster import KMeans
import numpy as np

seed = 123

# Load data
data_pd = load_data()
data_np = data_pd.to_numpy()
print("data_np: ", data_np)

# Normalization
points = min_max_nomalization(data_np)

num_trials = 100
# k = 8
k = 3

# model = K_Means(random_seed=seed)
# k_means_centroids = model.fit(points, k, num_trials)
# closest_clusters = model.closest_centroid()

kmeans = KMeans(n_clusters=k, random_state=seed, n_init=num_trials, max_iter=1000).fit(points)

closest_clusters = kmeans.predict(points)

from evaluation import plots
plots.plot3D_numpy(data_np, 
                    axis_labels=["Temperature (°C)", "Humidity (%)", "CO Value (ppm)"], 
                    title="k-means centroids",
                    point_size=1,
                    color=closest_clusters,
                #    color=model.closest_centroid(),
                    color_map="viridis", 
                    show_colorbar=False)

# from sklearn.metrics import silhouette_score
# silhouette_avg = silhouette_score(points, closest_clusters)

# print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
# plots.plot_Silhouette(x=points, y=closest_clusters, silhouette_avg=silhouette_avg)
