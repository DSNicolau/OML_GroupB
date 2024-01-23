import sys

sys.path.append("Python/clustering/")

from utils import utils
import skfuzzy as fuzz
import numpy as np
from sklearn.metrics import silhouette_score
import optuna
import os




def objective(trial):
    n_clusters = trial.suggest_int("n_clusters", 2, 100)
    m = trial.suggest_int("fuzzy parameter", 2, 12)
    error = trial.suggest_float("error", 1e-4, 0.01)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_np.transpose(), n_clusters, m, error=error, maxiter=1000, init=None
    )
    cluster_labels = np.argmax(u, axis=0)
    silhouette = silhouette_score(data_np, cluster_labels)
    variances = []
    for cluster_id in range(n_clusters):  # Assuming 5 clusters
        cluster_points = data_np[cluster_labels == cluster_id]
        centroid = np.mean(cluster_points, axis=0)
        variance = np.mean(np.sum((cluster_points - centroid) ** 2, axis=1))
        variances.append(variance)
    avg_variance = np.mean(variances)
    trial.report(
        avg_variance, step=0
    )
    trial.report(
        silhouette, step=1
    )

    return fpc

if __name__ == "__main__":
    data = utils.load_data()
    data = utils.min_max_nomalization_pandas(data)
    data_np = data.to_numpy()
    os.chdir("Python/clustering/examples/")
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///optuna_studies.db",
        study_name="optuna_fuzzy_clustering",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=1000)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/clustering/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
