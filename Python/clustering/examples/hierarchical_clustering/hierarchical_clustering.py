import sys

sys.path.append("Python/clustering/")

from utils import utils
import evaluation
from sklearn.cluster import AgglomerativeClustering

if __name__ == "__main__":
    data = utils.load_data()
    data_norm = utils.min_max_nomalization_pandas(data)
    data_np = data_norm.to_numpy()
    clustering = AgglomerativeClustering(n_clusters=8, linkage="average").fit(data_np)
    evaluation.plot3D(data=data, clusters=clustering.labels_)
