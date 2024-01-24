import sys

sys.path.append("Python/classification/")

from utils import utils
import evaluation
import optuna
import os
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as kNN

def objective(trial):

    train_data, train_label = utils.get_numpy_features(train, no_time=True)
    test_data, test_label = utils.get_numpy_features(test, no_time=True)

    # pca_components = trial.suggest_int("pca_components", 1, 9) # 9 is the number o features we originally have
    pca_components = trial.suggest_int("pca_components", 4, 5) # 9 is the number o features we originally have
    n_neighbours = trial.suggest_int("n_neighbours", 1, 500)
    p = trial.suggest_int("p", 1, 20)

    trial_number = trial.number

    pca = PCA(n_components=pca_components, random_state=0)
    pca.fit(train_data)
    # Transform train and test following the train set PCA
    train_data = pca.transform(train_data) 
    test_data = pca.transform(test_data)


    knn = kNN(n_neighbors=n_neighbours, p=p, metric="minkowski")
    knn.fit(train_data, train_label)
    predicts = knn.predict(test_data)

    cf_matrix = evaluation.confusion_matrix(test_label, predicts)
    evaluation.displayConfMatrix(
        cf_matrix, save_name="KNN/results/knn_neighbours_p_pca_{}.png".format(trial_number)
    )
    accuracy, precision, recall, f1_score = evaluation.evaluate(
        cf_matrix=cf_matrix
    )
    return accuracy, precision, recall, f1_score


if __name__ == "__main__":
    train, _, test = utils.load_data(normalize=True)
    
    # Change this path to the path you wish your database to be stored
    os.chdir("Python/classification/examples/")
    study = optuna.create_study(
        directions=["maximize", "maximize", "maximize", "maximize"],
        storage="sqlite:///optuna_studies.db",
        study_name="knn__neighboursp_pca",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=150)
    # To then visualize the results on the database:
    # please install optuna-dashboard (pip install optuna-dashboard)
    # move to the directory with the database (Python/classification/examples/)
    # run the command: optuna-dashboard sqlite:///optuna_studies.db
