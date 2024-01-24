import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go



def displayConfMatrix(cf_matrix, save_name=None):
    group_names = ["True Neg", "False Pos", "False Neg", "True Pos"]

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]

    for i in range(cf_matrix.shape[0]):
        for value in cf_matrix[i].flatten() / np.sum(cf_matrix[i]):
            group_percentages.append("{0:.2%}".format(value))

    labels = [
        f"{v1}\n{v2}\n{v3}"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]

    labels = np.asarray(labels).reshape(2, 2)

    plt.figure()
    ax = sns.heatmap(cf_matrix, annot=labels, fmt="", cmap="Blues")

    ax.set_title("Confusion Matrix\n\n")
    ax.set_xlabel("\nPredicted Values")
    ax.set_ylabel("Actual Values ")

    ## Ticket labels
    ax.xaxis.set_ticklabels(["No Motion", "Motion"])
    ax.yaxis.set_ticklabels(["No Motion", "Motion"])
    plt.gcf().set_size_inches(8, 6)
    ## Display the visualization of the Confusion Matrix.
    if save_name:
        figure_name = save_name
        plt.savefig(figure_name)
    else:
        plt.show()


def displayHeatMap(
    matrix,
    x_label=None,
    y_label=None,
    x_tick_labels=None,
    y_tick_labels=None,
    title="Heat Map",
    save_name=None,
):
    labels = [str(i) for i in matrix.flatten()]
    labels = np.asarray(labels).reshape(matrix.shape)
    ax = sns.heatmap(matrix, annot=labels, fmt="", cmap="Blues")
    ax.set_title(title + "\n\n")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if x_tick_labels:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels:
        ax.set_yticklabels(y_tick_labels)

    if save_name:
        figure_name = save_name
        plt.savefig(figure_name)
    else:
        plt.show()


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
                    colorbar=dict(thickness=40, title="Labels"),
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