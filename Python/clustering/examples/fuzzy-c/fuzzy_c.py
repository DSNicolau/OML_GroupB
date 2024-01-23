import sys

sys.path.append("Python/clustering/")

from utils import utils
import evaluation
import skfuzzy as fuzz

if __name__ == "__main__":
    data = utils.load_data()
    data_norm = utils.min_max_nomalization_pandas(data)
    data_np = data_norm.to_numpy()
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
        data_np.transpose(), 11, 2, error=0.005, maxiter=1000, init=None
    )
    print(fpc)
    evaluation.plot3D(data=data, clusters=u.argmax(axis=0))
