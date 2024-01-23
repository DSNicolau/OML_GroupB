import sys

sys.path.append("Python/classification/")

import optuna
import os

def objective(trial):
    n_neighbours = trial.suggest_int("n_neighbours", 1, 500)
    p = trial.suggest_int("p", 1, 20)
    if prune:
        raise optuna.TrialPruned()
    return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    os.chdir("Python/classification/examples/")
    or_study = optuna.load_study(storage="sqlite:///optuna_studies.db",study_name="svm_poly_rbf_sigmoid_study",)
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"],
        storage="sqlite:///optuna_study.db",
        study_name="svm_poly_rbf_sigmoid_study",
        load_if_exists=True,
    )
    for tri in or_study.trials:
        prune = False
        try:
            accuracy, precision, recall, f1_score, _ = tri.values
        except:
            prune = True
        study.enqueue_trial(tri.params)
        study.optimize(objective, n_trials=1)

