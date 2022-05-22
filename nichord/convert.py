import numpy as np
from typing import Tuple


def convert_matrix(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts your adjacency (connectivity) matrix into a list of edges (i, j) 
        and their weights
    :param adj: the matrix
    """
    idxs = np.tril_indices(adj.shape[0], k=-1)
    weights = adj[idxs]
    idxs = np.array(idxs).T
    return idxs, weights
