import sys

sys.path.append("Python/clustering/")

from utils import utils
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import optuna
import os


def objective(trial):
    n_clusters = trial.suggest_int("n_clusters", 1, 200)
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(data_np)
    cluster_labels = clustering.labels_
    variances = []
    for cluster_id in range(n_clusters):  # Assuming 5 clusters
        cluster_points = data_np[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        variance = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
        variances.append(variance)
    avg_variance = np.mean(variances)
    return avg_variance


if __name__ == "__main__":
    data = utils.load_data()
    data = utils.min_max_nomalization_pandas(data)
    data_np = data.to_numpy()
    os.chdir("Python/clustering/examples/")
    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_studies.db",
        study_name="optuna_hierarchical_clustering",
        load_if_exists=True,
    )
    j = 0
    for i in range(1, 201):
        study.enqueue_trial({"n_clusters": i})
        j += 1
    study.optimize(objective, n_trials=j)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/clustering/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
