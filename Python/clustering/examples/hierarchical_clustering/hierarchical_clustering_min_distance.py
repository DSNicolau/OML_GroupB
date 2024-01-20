import sys

sys.path.append("Python/clustering/")

from utils import utils
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from evaluation import plot_Scatter
from sklearn.metrics import silhouette_score


def hierarchical_clustering(n_clusters, data_np):
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data_np)
    cluster_labels = clustering.labels_
    silhouette = 0
    if n_clusters > 1:
        silhouette = silhouette_score(data_np, cluster_labels)
    variances = []
    for cluster_id in range(n_clusters):  # Assuming 5 clusters
        cluster_points = data_np[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        variance = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
        variances.append(variance)
    avg_variance = np.mean(variances)
    return avg_variance, silhouette


def min_derivative_distance(variances):
    derivatives = np.array(
        [variances[i] - variances[i + 1] for i in range(len(variances) - 1)]
    )
    abs_derivatives = np.abs(derivatives)
    return np.argmin(abs_derivatives) + 1


def eucldean_distance(variances, x):
    score = np.sqrt(variances**2 + (x**2))
    return score


if __name__ == "__main__":
    data = utils.load_data()
    data = utils.min_max_nomalization_pandas(data)
    data_np = data.to_numpy()
    max_trials = 100
    var_silh = []
    for n_clusters in range(1, max_trials + 1):
        var_silh.append(hierarchical_clustering(n_clusters=n_clusters, data_np=data_np))
    var_silh = np.array(var_silh)
    variances = var_silh[:, 0]
    silhouettes = var_silh[:, 1]
    normalized_variations = variances / np.max(variances)
    best_index = min_derivative_distance(normalized_variations)
    score = eucldean_distance(
        variances=normalized_variations,
        x=np.linspace(1, max_trials, max_trials) / best_index,
    )

    # Define the colors (red is the best)
    colors = np.zeros_like(silhouettes)
    colors[np.argmax(silhouettes)] = 1

    # Plot Silhouette Score
    plot_Scatter(
        x=np.linspace(1, max_trials, max_trials),
        y=silhouettes,
        xlabel="Number of Clusters",
        ylabel="Silhouette Score",
        colors=colors,
    )

    # Define the colors (red is the best)
    colors = np.zeros_like(score)
    colors[np.argmin(score)] = 1

    # Plot the results
    plot_Scatter(
        x=np.linspace(1, max_trials, max_trials),
        y=variances,
        xlabel="Number of Clusters",
        ylabel="Variation Score",
        colors=colors,
    )
    # PLot the Euclidean distances
    plot_Scatter(
        x=np.linspace(1, max_trials, max_trials),
        y=score,
        xlabel="Number of Clusters",
        ylabel="Euclidean Distance",
        colors=colors,
    )
