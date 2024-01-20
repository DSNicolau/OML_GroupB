import plotly.graph_objects as go
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot3D(data, clusters=[np.nan, np.nan]):
    if all(np.isnan(clusters)):
        clusters = np.zeros(len(data))
    if len(clusters) != len(data):
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


def plot_Scatter(x, y, xlabel, ylabel, colors, title=""):
    fig = go.Figure(
        data=go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=8,
                color=colors,  # set color to an array/list of desired values
                colorscale="Bluered",  # choose a colorscale
            )
            if not np.all(colors == 0)
            else dict(
                size=8,
                color="blue",  # set color to a single value
            ),
            line=dict(color="navy", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="CMU Serif Extra", size=35),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        color="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="gainsboro",
        zeroline=True,
        zerolinecolor="black",
    )
    fig.update_yaxes(
        color="black",
        zeroline=True,
        zerolinecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="gainsboro",
    )

    fig.show()


def plot_Dendogram(data, color_threshold, title=""):
    fig = ff.create_dendrogram(data, color_threshold=color_threshold)
    fig.update_layout(
        title=title,
        font=dict(family="CMU Serif Extra", size=35),
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        color="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="gainsboro",
        zeroline=True,
        zerolinecolor="black",
    )
    fig.update_yaxes(
        color="black",
        zeroline=True,
        zerolinecolor="black",
        showgrid=True,
        gridwidth=1,
        gridcolor="gainsboro",
    )

    fig.show()


def plot_Silhouette(
    x,
    y,
    silhouette_avg,
    xlabel="Silhouette Coefficient Values",
    ylabel="Cluster Label",
    title="",
):
    from sklearn.metrics import silhouette_samples

    fig, ax = plt.subplots()
    n_clusters = np.max(y) + 1

    sample_silhouette_values = silhouette_samples(x, y)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.grid(axis="x")

    ax.set_yticks([])
    ax.set_xticks([-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    plt.show()
