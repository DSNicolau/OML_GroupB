import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


distances = {
    "euclidean": euclidean_distance,
    "manhattan": manhattan_distance
}

def distance(x1, x2, distance_type="euclidean"):
    if x1.shape != x2.shape:
        raise ValueError("The shape of of both arrays must be the same!")
    if distance_type not in distances:
        raise ValueError("Distance type not supported!")
    return distances[distance_type](x1, x2)