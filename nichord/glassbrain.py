import numpy as np
import matplotlib.pyplot as plt
from nilearn import plotting
import matplotlib
from typing import Union


def plot_glassbrain(idx_to_label: dict, edges: list,
                    edge_weights: Union[list, None],
                    fp_glass: Union[None, str],
                    coords: list,
                    cmap: Union[None, str, matplotlib.colors.Colormap] = None,
                    node_size: Union[None, float, int] = 5,
                    linewidths: Union[None, float, int, list] = 6,
                    alphas: Union[None, float, int, list] = None,
                    network_colors: Union[None, dict] = None,
                    network_order: Union[list, None] = None,) -> None:
    """
    Plots the connectome on a brain, where the node colors correspond to the
        ROI's label (specified by idx_to_label). Each edge's color correspond
        to the edge's weight or is set to gray if all weighst are equal to 1.
        Can either be used to save a file or open the connectome in your
        browser, depending on whether fp_out is None.

    :param idx_to_label: dict mapping each ROI index to its chord label
        (e.g., {0: "FPCN"})
    :param edges: list of tuples (a, b), where a = index of the 1st ROI &
        b = index of the 2nd ROI
    :param edge_weights: list of edge weights. The nth weight, here, should
        correspond to the nth edge in edges, above. If None or all the weights
        are equal to 1, then grey edges will be plotted.
    :param fp_glass: filepath where the connectome glass brain will be saved.
        If None, then the connectome will be loaded in your browser.
    :param coords: list of coordinates
    :param cmap: can either be None (default will be used), a string (will
        retrieve cmap from matplotlib), or a matplotlib colormap
    :param node_size: Size of the nodes (all must be constant)
    :param linewidths: linewidths used for the edges
    :param alphas: alphas used for the edges
    :param network_colors: dict mapping each label to a matplotlib color. If
        None, then default colors will be set based on matplotlib.
    :param network_order: If network_colors is None, then this list
        will specify the order in which default matplotlib colors will be
        assigned to networks.
    """
    if network_colors is None:
        if network_order is None:
            network_order = sorted(list(set(idx_to_label.values())))
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        network_colors = dict((network, color) for network, color in
                              zip(network_order,
                                  default_colors[:len(network_order)]))

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if edge_weights is None:
        edge_weights = [1]*len(edges)

    nonzero_node_idxs = list(set(node_idx for tup in edges for node_idx in tup))
    n = len(nonzero_node_idxs)
    adj = np.zeros((n, n))
    tril_idx = np.tril_indices(n, k=-1)
    if all(weight == 1 for weight in edge_weights):
        adj[tril_idx] = adj.T[tril_idx] = 0.5
        if cmap is None: cmap = plt.get_cmap('Greys')
        edge_alphas = .4
    else:
        node_idx_to_i = dict(
            (node_idx, i) for i, node_idx in enumerate(nonzero_node_idxs))
        for edge, edge_weight in zip(edges, edge_weights):
            adj[(node_idx_to_i[edge[0]], node_idx_to_i[edge[1]])] = edge_weight
            adj[(node_idx_to_i[edge[1]], node_idx_to_i[edge[0]])] = edge_weight
        if cmap is None: cmap = plt.get_cmap('Spectral_r')
        edge_alphas = .9

    node_coords_pruned = [coords[i] for i in nonzero_node_idxs]
    node_colors = [network_colors[idx_to_label[node_i]] for node_i in
                   nonzero_node_idxs]
    if fp_glass is None:
        cv = plotting.view_connectome(adj, node_coords_pruned, edge_cmap=cmap,
                                      node_color=node_colors,
                                      node_size=node_size,
                                      linewidth=linewidths, colorbar=False)
        cv.open_in_browser()
    else:
        plotted_img = \
            plotting.plot_connectome(adj, node_coords_pruned,
                                    edge_cmap=cmap, node_color=node_colors,
                                    node_size=node_size, edge_kwargs=
                                    {'linewidth': linewidths/5,
                                     'alpha': edge_alphas if alphas is None
                                                     else alphas},
                                    colorbar=False)
        plotted_img.savefig(fp_glass, dpi=400)
        plotted_img.close()
