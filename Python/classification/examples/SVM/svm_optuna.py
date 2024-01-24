import sys

sys.path.append("Python/classification/")

from utils import utils
from sklearn import svm
import evaluation
import os
import optuna


def objective(trial):
    c = trial.suggest_float("C", 1e-2, 1e3, log=True)
    kernel = trial.suggest_categorical("kernel", ["poly", "rbf", "sigmoid"])
    gamma = trial.suggest_float("gamma", 1e-6, 1e-2, log=True)

    # Create model and predict
    clf = svm.SVC(kernel=kernel, C=c, gamma=gamma)
    clf.fit(train_data, train_label)
    predicts = clf.predict(test_data)

    trial_number = trial.number
    # Get the confusion matrix and save it
    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(
        cf_matrix, save_name="SVM/results/svm_poly_rbf_sigmoid_{}.png".format(trial_number)
    )

    # Evaluate
    accuracy, precision, recall, f1_score = evaluation.evaluate(
        cf_matrix=cf_matrix
    )
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    train, _, test = utils.load_data(normalize=True)
    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)
    # Change this path to the path you wish your database to be stored
    os.chdir("Python/classification/examples/")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"],
        storage="sqlite:///optuna_studies.db",
        study_name="svm_poly_rbf_sigmoid_study",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=60)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/classification/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
