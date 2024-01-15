import sys
sys.path.append('Python/')
from clustering.models.k_means import K_Means
from clustering.utils.utils import load_data, min_max_nomalization

import optuna

seed = 123

# Load data
data_pd = load_data()
data_np = data_pd.to_numpy()
print("data_np: ", data_np)

# Normalization
points = min_max_nomalization(data_np)

# trial_num_trials = 10
# trial_k = 3

# model = K_Means(random_seed=seed)
# k_means_centroids = model.fit(points, trial_k, trial_num_trials)
# k_means_variances = model.trial_variances
# min_variance = min(k_means_variances)

from sklearn.cluster import KMeans
import numpy as np

def objective_v2(trial):
    trial_num_trials = 100
    trial_k = trial.suggest_int('k', 1, 100, log=False)
    
    kmeans = KMeans(n_clusters=trial_k, random_state=0, n_init="auto", max_iter=trial_num_trials).fit(points)
    
    closest_clusters = kmeans.predict(points)
    clusters = kmeans.cluster_centers_
    distances = [np.sqrt(((points[closest_clusters==i] - clusters[i])**2).sum(axis=1)) for i in range(clusters.shape[0])]
    variances = [np.var(distances[i], axis=0) for i in range(clusters.shape[0])]
    total_variance = np.sum(variances)/clusters.shape[0]
    return total_variance

def objective(trial):
    # trial_num_trials = trial.suggest_int('num_trials', 3, 30, log=False)
    trial_num_trials = 10
    trial_k = trial.suggest_int('k', 1, 100, log=False)

    model = K_Means(random_seed=seed)
    k_means_centroids = model.fit(points, trial_k, trial_num_trials)
    k_means_variances = model.trial_variances
    min_variance = min(k_means_variances)
    return min_variance

# studyName = "OML_K_Means_k_study_Test_vScikitLearn_5"
studyName = "OML_K_Means_k_study_Test_v2"

study = optuna.create_study(
                            # directions=['maximize', 'maximize'],
                            direction='minimize',
                            # storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

for i in range(1, 101):
    study.enqueue_trial({"k": i})

study.optimize(objective, n_trials=100)

# variations = 1

# d_variations_dk = np.diff(variations)
# d_variations_dk*variations