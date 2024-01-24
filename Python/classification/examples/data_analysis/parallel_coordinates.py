import optuna
import os

os.chdir("Python/classification/examples/")
import plotly.express as px
import pandas as pd


if __name__ == "__main__":
    study = optuna.load_study(
        storage="sqlite:///optuna_studies.db",
        study_name="knn__neighboursp_pca",
    )

    values_dic = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Neighbours": [],
        "p": [],
        "PCA Components": [],
        "Average": [],
    }
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.PRUNED:
            continue
        values_dic["Accuracy"].append(trial.values[0])
        values_dic["Precision"].append(trial.values[1])
        values_dic["Recall"].append(trial.values[2])
        values_dic["F1"].append(trial.values[3])
        values_dic["Neighbours"].append(trial.params["n_neighbours"])
        values_dic["p"].append(trial.params["p"])
        values_dic["PCA Components"].append(trial.params["pca_components"])
        values_dic["Average"].append(
            (trial.values[0] + trial.values[1] + trial.values[2] + trial.values[3]) / 4
        )

    df = pd.DataFrame(values_dic)
    fig = px.parallel_coordinates(
        df, color="Average", color_continuous_scale=px.colors.sequential.Reds,
        dimensions=["Accuracy", "Precision", "Recall", "F1", "Neighbours", "p", "PCA Components"],
    )

    fig.show()
