import numpy as np
import plotly.graph_objects as go

import optuna 

studyName = "OML_K_Means_NumTrials_study"

study = optuna.create_study(
                            directions=['minimize', 'maximize'],
                            # direction='minimize',
                            # storage="sqlite:////nfs/home/nvasconcellos.it/softLinkTests/xDNN_test.db",
                            storage="sqlite:///OML_Database.db",
                            study_name=studyName, load_if_exists=True)

trials_df = study.trials_dataframe()
trials_df = trials_df[trials_df["state"] == "COMPLETE"]
print(trials_df)
x = trials_df["params_k"][1:]
num_trials = trials_df["values_0"][1:]

mean_num_trials = np.mean(num_trials)

# Create scatter plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0, 60],
    y=[mean_num_trials, mean_num_trials],
    mode='lines', 
    line=dict(color="orange", width=5)
))
fig.add_trace(go.Scatter(
    # x=x[1:],
    x=x,
    y=num_trials,    
    # mode='lines+markers',
    mode='markers',
    # marker=dict(
    #     size=8,
    #     color=num_trials,                # set color to an array/list of desired values
    #     colorscale="Bluered",   # choose a colorscale
    #     colorbar=dict(title="Variation Score"),
    #     showscale=True
    # ),
    marker=dict(
        size=8,
        color='navy'   # set color to a single value
    ),
    line=dict(color='navy', width=2)
))


fig.update_layout(
    title="",
    xaxis_title="Number of Clusters",
    # yaxis_title="Execution Time (s)",
    # yaxis_title="Number of Prototypes",
    yaxis_title="Number of Trials",
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