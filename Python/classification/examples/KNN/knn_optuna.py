import sys

sys.path.append("Python/classification/")

from utils import utils
import models
import evaluation
import optuna
import os


def objective(trial):
    n_neighbours = trial.suggest_int("n_neighbours", 1, len(test_label))
    p = trial.suggest_int("p", 1, 20)
    trial_number = trial.number
    predicts = knn.predict(test_data, n_neighbours=n_neighbours, distance_type=p)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(
        cf_matrix, save_name="examples/KNN/results/knn_{}.png".format(trial_number)
    )
    return evaluation.evaluate(cf_matrix=cf_matrix)


if __name__ == "__main__":
    train, _, test = utils.load_data()
    knn = models.kNN()
    train_data, train_label = utils.get_numpy_features(train)
    knn.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test)
    # Change this path to the path you wish your database to be stored
    os.chdir("Python/classification/examples/")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize", "maximize"],
        storage="sqlite:///optuna_studies.db",
        study_name="knn_neighbours_p_study",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/classification/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
