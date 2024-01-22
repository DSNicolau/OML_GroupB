import sys

sys.path.append("Python/clustering/")

from utils import utils
from sklearn.cluster import AgglomerativeClustering
from evaluation import plot_Silhouette
from sklearn.metrics import silhouette_score


if __name__ == "__main__":
    data = utils.load_data()
    data = utils.min_max_nomalization_pandas(data)
    data_np = data.to_numpy()
    n_clusters = 3
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data_np)
    cluster_labels = clustering.labels_
    silhouette_avg = silhouette_score(data_np, cluster_labels)

    # Plot Silhouette Plot

    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    plot_Silhouette(x=data_np, y=cluster_labels, silhouette_avg=silhouette_avg)

