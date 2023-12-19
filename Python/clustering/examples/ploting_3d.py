import sys

sys.path.append("Python/clustering/")

from utils import utils
import evaluation

if __name__ == "__main__":
    data = utils.load_data()
    evaluation.plot3D(data=data)