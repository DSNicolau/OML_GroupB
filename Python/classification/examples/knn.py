import sys

sys.path.append("Python/classification/")

import utils
import models
import evaluation


if __name__ == "__main__":
    train, val, test = utils.load_data(fill_method="ffill")
    knn = models.kNN()
    knn.fit(train[0][:, 5:], train[1])
    predicts = knn.predict(test[0][:, 5:], 1)
    cf_matrix = evaluation.confusion_matrix(test[1], predicts)
    evaluation.displayConfMatrix(
        cf_matrix,
        save_name="/home/danielnicolau/Documents/GitHub/OML_GroupB/confusion_matrix_knn1.png",
    )
