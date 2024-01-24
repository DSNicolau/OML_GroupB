import optuna
import os

os.chdir("Python/classification/examples/")
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go


if __name__ == "__main__":
    study = optuna.load_study(
        storage="sqlite:///optuna_studies.db",
        study_name="knn_pca",
    )

    values_dic = {
        "Accuracy": [],
        "F1 Score": [],
        "PCA Components": [],
    }
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.FAIL or trial.state == optuna.trial.TrialState.PRUNED:
            continue
        values_dic["Accuracy"].append(trial.values[0])
        values_dic["F1 Score"].append(trial.values[3])
        values_dic["PCA Components"].append(trial.params["pca_components"])


    df = pd.DataFrame(values_dic)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=values_dic["PCA Components"], y=values_dic["Accuracy"],
                    mode='lines+markers',
                    name='Accuracy'))

    fig.add_trace(go.Scatter(x=values_dic["PCA Components"], y=values_dic["F1 Score"],
                    mode='lines+markers',
                    name='F1 Score'))

    fig.update_layout(xaxis_title='PCA Components',
                   yaxis_title='Accuracy/F1 Score')
    fig.show()
