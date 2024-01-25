import numpy as np
import plotly.graph_objects as go

import optuna 

studyName = "OML_K_Means_k_study"

study = optuna.create_study(
                            direction='minimize',
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

trials_df = study.trials_dataframe()
print(trials_df)
x = trials_df["params_k"][trials_df["state"] == "COMPLETE"]
variations = trials_df["value"][trials_df["state"] == "COMPLETE"]

def find_best_k(y, x):
    d_y_dk = np.diff(y)
    d_y_dk_abs = np.abs(d_y_dk)
    x_limit = np.argmin(d_y_dk_abs) + 1
    y_limit = np.max(y)
    # Print Limits
    print("x_limit:", x_limit)
    print("y_limit:", y_limit)
    x_normalized = x/x_limit
    y_normalized = y/y_limit
    score = np.sqrt(x_normalized**2 + y_normalized**2)
    return score

score = find_best_k(variations, x)
best = np.argmin(score)
colors = np.zeros_like(score)
colors[best] = 1
# Create scatter plot
fig = go.Figure(data=go.Scatter(
    x=x,
    y=score,    
    mode='markers',
    marker=dict(
        size=8,
        color=colors,                # set color to an array/list of desired values
        colorscale="Bluered",   # choose a colorscale
        colorbar=dict(title="Variation Score"),
        showscale=False
    ),
    line=dict(color='navy', width=2)
))

fig.update_layout(
    title="",
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Euclidean Error",
    font=dict(
        family='CMU Serif Extra',
        size=35
    ), 
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.update_xaxes(color = 'black', showgrid=True, gridwidth=1, gridcolor='gainsboro', 
                 zeroline=True, zerolinecolor='black')
fig.update_yaxes(color = 'black', zeroline=True, zerolinecolor='black', showgrid=True, gridwidth=1, gridcolor='gainsboro')


fig.show()