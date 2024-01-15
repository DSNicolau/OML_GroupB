import numpy as np
import plotly.graph_objects as go

import optuna 

studyName = "OML_K_Means_k_study_Test_v2"

study = optuna.create_study(
                            # directions=['maximize', 'maximize'],
                            direction='minimize',
                            # storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

trials_df = study.trials_dataframe()
print(trials_df)
x = trials_df["params_k"]
variations = trials_df["value"]

x = x/68
variations = variations/np.max(variations)

def find_best_k(variations, x, x_0, lamda):
    d_variations_dk = np.diff(variations)
    # score = (1/variations[1:])*(-d_variations_dk)*np.exp((-(x[1:]-x_0)/lamda))
    score = np.sqrt(variations**2 + (x**2))
    return score

lamda = 15
num_min_clusters = 5
score = find_best_k(variations, x, num_min_clusters, lamda)
best = np.argmin(score)
colors = np.zeros_like(score)
colors[best] = 1
# Create scatter plot
fig = go.Figure(data=go.Scatter(
    # x=x[1:],
    x=x*68,
    y=score,    
    # mode='lines+markers',
    mode='markers',
    marker=dict(
        size=8,
        color=colors,                # set color to an array/list of desired values
        colorscale="Bluered",   # choose a colorscale
        colorbar=dict(title="Variation Score"),
        showscale=True
    ),
    # marker=dict(
    #     size=8,
    #     color='navy'   # set color to a single value
    # ),
    line=dict(color='navy', width=2)
))

fig.update_layout(
    title="",
    xaxis_title="Number of Clusters",
    # yaxis_title="Execution Time (s)",
    # yaxis_title="Number of Prototypes",
    yaxis_title="Score",
    font=dict(
        # family="'CMU Serif Extra', sans-serif",
        family='CMU Serif Extra',
        size=35
    ), 
    plot_bgcolor='rgba(0,0,0,0)'
)

fig.update_xaxes(color = 'black', showgrid=True, gridwidth=1, gridcolor='gainsboro', 
                 zeroline=True, zerolinecolor='black')
fig.update_yaxes(color = 'black', zeroline=True, zerolinecolor='black', showgrid=True, gridwidth=1, gridcolor='gainsboro')

# fig.update_yaxes(ticks="outside", tickwidth=4, tickcolor='black', ticklen=20, nticks=5)

fig.show()