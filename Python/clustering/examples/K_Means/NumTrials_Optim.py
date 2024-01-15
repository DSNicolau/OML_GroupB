import sys
sys.path.append('Python/')
from clustering.utils.utils import load_data, min_max_nomalization

import optuna

from sklearn.cluster import KMeans
import numpy as np

# Load data
data_pd = load_data()
data_np = data_pd.to_numpy()
print("data_np: ", data_np)

# Normalization
points = min_max_nomalization(data_np)


max_iter = 300
seeds = [123, 234, 345, 456, 567]

def objective(trial):
    trial_k = trial.suggest_int('k', 1, 100, log=False)
    
    n_star_set = []
    
    for seed in seeds:
        kmeans_opt = KMeans(n_clusters=trial_k, init='k-means++', n_init=max_iter, random_state=seed, max_iter=100000).fit(points)  
        optimal_clusters = kmeans_opt.cluster_centers_.round(decimals=10)
        
        S_set = [max_iter]
        I_set = [0]
        
        i = 1
        while True:   
            print("S_set: ", S_set)
            print("I_set: ", I_set)     
            n = int((min(S_set) + max(I_set)) / 2)
            print("%d. n = %d " % (i, n))
            i += 1
            if n == max(I_set):
                break
            kmeans = KMeans(n_clusters=trial_k, init='k-means++', n_init=n, random_state=seed, max_iter=100000).fit(points)    
            clusters = kmeans.cluster_centers_.round(decimals=10)
            if (clusters != optimal_clusters).any():
                print("Inferior!")
                print("Number of Differents: %d/%d" % ((clusters != optimal_clusters).sum(), clusters.shape[0] * clusters.shape[1]))
                I_set.append(n)
            else:
                print("Superior!")
                S_set.append(n)

        n_star = min(S_set)
        n_star_set.append(n_star)
        
    optimal_n = max(n_star_set)
    print("n_star_set: ", n_star_set)
    support = (np.array(n_star_set) == optimal_n).sum()
    return optimal_n, support

studyName = "OML_K_Means_NumTrials_study"

study = optuna.create_study(
                            directions=['minimize', 'maximize'],
                            # direction='minimize',
                            # storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

# for i in range(1, 101):
#     study.enqueue_trial({"k": i})

study.optimize(objective, n_trials=50)