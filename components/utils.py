import numpy as np


def min_max_normalize(arr):

    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr

def l2_normalize(arr):
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm

