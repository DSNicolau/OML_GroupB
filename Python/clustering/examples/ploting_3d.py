import sys

sys.path.append("Python/clustering/")

from utils import utils
import evaluation
from sklearn.cluster import AgglomerativeClustering

if __name__ == "__main__":
    data = utils.load_data()
    for key in data.keys():
        data[key] = (data[key]-data[key].min())/(data[key].max()-data[key].min())
    data_np = data.to_numpy()
    clustering = AgglomerativeClustering(n_clusters=5).fit(data_np)
    evaluation.plot3D(data=data, clusters=clustering.labels_)

    