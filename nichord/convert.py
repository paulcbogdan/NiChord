import numpy as np
from typing import Union, Tuple


def convert_matrix(adj: Union[list, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts your adjacency (connectivity) matrix into a list of edges (i, j) 
        and their weights
    :param adj: the matrix
    """
    if isinstance(adj, list):
        adj = np.array(adj)
    idxs = np.tril_indices(adj.shape[0], k=-1)
    weights = adj[idxs]
    idxs = np.array(idxs).T
    smol = 1e-6
    idxs = idxs[(weights > smol) | (weights < -smol)]
    weights = weights[(weights > smol) | (weights < -smol)]
    return idxs, weights
