import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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
