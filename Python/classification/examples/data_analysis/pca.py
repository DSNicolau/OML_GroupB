import sys

sys.path.append("Python/classification/")

from utils import utils
from sklearn.decomposition import PCA
from evaluation import plot3D
import pandas as pd


if __name__ == "__main__":
    train, val, test = utils.load_data(normalize=True)
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)
    pca = PCA(n_components=3)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    print(train_data.shape)
    print(test_data.shape)

    train_pd = pd.DataFrame(train_data, columns=["PC1", "PC2", "PC3"])


    plot3D(train_pd, train_label)


