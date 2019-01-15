import numpy as np

def euclidean_distance(x1, x2):
    diff_squared = (x1 - x2)**2
    distance_squared = diff_squared.sum()
    return np.sqrt(distance_squared)
