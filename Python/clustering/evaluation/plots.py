import plotly.graph_objects as go
import numpy as np


def plot3D(data, clusters=[np.nan, np.nan]):
    if all(np.isnan(clusters)):
        clusters = np.zeros(len(data))
    if len(clusters)!= len(data):
        raise ValueError("Data and clusters must be of same length")
    labels = data.keys()
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=data[labels[0]],
                y=data[labels[1]],
                z=data[labels[2]],
                mode="markers",
                marker=dict(
                    size=1,
                    color=clusters,
                    colorscale="Viridis",
                    opacity=0.8,
                    colorbar=dict(thickness=40, title="Clustering Data"),
                ),
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=labels[0],
            yaxis_title=labels[1],
            zaxis_title=labels[2],
        )
    )
    fig.show()
