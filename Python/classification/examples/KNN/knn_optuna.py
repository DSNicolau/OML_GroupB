import sys

sys.path.append("Python/classification/")

from utils import utils
import models
import evaluation
import optuna
import os


def objective(trial):
    n_neighbours = trial.suggest_int("n_neighbours", 1, 500)
    p = trial.suggest_int("p", 1, 20)
    trial_number = trial.number
    predicts = knn.predict(test_data, n_neighbours=n_neighbours, distance_type=p)
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(
        cf_matrix, save_name="KNN/results/knn_filt_butter_{}.png".format(trial_number)
    )
    accuracy, precision, recall, f1_score = evaluation.evaluate(
        cf_matrix=cf_matrix
    )
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    train, _, test = utils.load_data(normalize=True, filt="butter")
    knn = models.kNN()
    train_data, train_label = utils.get_numpy_features(train, no_time=False)
    knn.fit(train_data, train_label)
    test_data, test_label = utils.get_numpy_features(test, no_time=False)
    # Change this path to the path you wish your database to be stored
    os.chdir("Python/classification/examples/")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"],
        storage="sqlite:///optuna_studies.db",
        study_name="knn_neighbours_p_study_filter_butterworth",
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=40)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/classification/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
