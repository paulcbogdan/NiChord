import math
import warnings

import matplotlib
from matplotlib import patches
from nilearn._utils.param_validation import check_threshold
from scipy import interpolate

from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from math import radians
from typing import Union, Tuple
from collections import OrderedDict

from scipy.stats import scoreatpercentile


class ROI_to_degree:
    """
    Used to calculate an ROI's position within the rim/label.
    Within the rim/label, ROIs are positioned based on their y coordinate
        (i.e., how anterior/posterior it is). If no coordinate is provided,
        the ROIs are positioned randomly (see plot_chord_diagram(...)).
    """

    def __init__(self, coords, idx_to_label, network_low_high, network_counts):
        self.idx_to_degree = {}
        network_ROI_degree_running_sum = defaultdict(lambda: 0)
        idx_coord_sorted_by_y = sorted(enumerate(coords), key=lambda x: x[1][1])
        for idx, coord in idx_coord_sorted_by_y:
            network = idx_to_label[idx]
            network_low, network_high = network_low_high[network]
            idx_degree = network_low + \
                         (network_ROI_degree_running_sum[network] + 0.5) / \
                         network_counts[network] * \
                         (network_high - network_low)
            self.idx_to_degree[idx] = idx_degree
            network_ROI_degree_running_sum[network] += 1

    def get_degree(self, idx):
        return self.idx_to_degree[idx]


def _network_order_colors_check(network_order, network_colors, idx_to_label):
    if network_order is None:
        network_order = sorted(list(set(idx_to_label.values())))
    else:
        idx_labels = set(idx_to_label.values())
        network_order = [network for network in network_order if
                         network in idx_labels]
        if idx_labels != set(network_order):
            missing_labels = idx_labels - set(network_order)
            raise ValueError('network_order, if specified, must contain all '
                             f'idx_to_label labels. '
                             f'Missing labels: {missing_labels}')

    if network_colors is None:
        default_colors = []
        for i in range(1 + len(network_order) // 10):
            default_colors += plt.rcParams['axes.prop_cycle'].by_key()['color']
        network_colors = dict((network, color) for network, color in
                              zip(network_order, default_colors[:len(network_order)]))
    else:
        idx_labels = set(idx_to_label.values())
        if not idx_labels.issubset(set(network_colors.keys())):
            missing_labels = set(idx_labels) - set(network_colors.keys())
            raise ValueError('network_colors, if specified, must describe all '
                             f'idx_to_label labels. '
                             f'Missing labels: {missing_labels}')
    return network_order, network_colors


def plot_chord(idx_to_label: dict,
               edges: Union[list, np.ndarray],
               fp_chord: Union[str, None] = None,
               edge_weights: Union[list, np.ndarray, None] = None,
               network_order: Union[list, None] = None,
               network_colors: Union[dict, None] = None,
               colors: Union[None, str, tuple, list] = None,
               linewidths: Union[None, float, int, list] = None,
               alphas: Union[None, float, int, list] = None,
               cmap: Union[None, str, matplotlib.colors.Colormap] = None,
               coords: Union[list, np.ndarray] = None,
               arc_setting: bool = True,
               cbar: Union[None, plt.colorbar] = None,
               do_ROI_circles: bool = False,
               do_ROI_circles_specific: bool = True,
               ROI_circle_radius: float = 0.005,
               black_BG: bool = False,
               label_fontsize: int = 60,
               do_monkeypatch: bool = True,
               vmin: Union[None, int, float] = None,
               vmax: Union[None, int, float] = None,
               plot_count: bool = False,
               plot_abs_sum: bool = False,
               norm_thickness: bool = False,
               edge_threshold: Union[float, int, str] = 0.,
               dpi: int = 400) -> None:
    """
    Plots the chord diagram and either saves a file if fp_chord is not None or
        opens it in a matplotlib window. For most of the arguments in these
        function, if None is passed, then a default setting is generated and
        used.

    :param idx_to_label: dict mapping each ROI index to its chord label
        (e.g., {0: "FPCN"})
    :param edges: list of tuples (a, b), where a = index of the 1st ROI &
        b = index of the 2nd ROI
    :param fp_chord: filepath to save the chord diagram. If None, a matplotlib
        window will open with the diagram
    :param edge_weights: list of edge weights. The nth weight, here, should
        correspond to the nth edge in edges, above
    :param network_order: list specifying the order of the labels (rims)
    :param network_colors: dict mapping each label to a matplotlib color
    :param linewidths: Value specifying the width of the edge arcs.
        If int or float, then all arcs will be the same width. If a list,
        then each arc will be set based on its list entry. If None, then
        widths will be generated based on the edge's weight.
    :param cmap: can either be None (default will be used), a string (will
        retrieve cmap from matplotlib), or a matplotlib colormap
    :param coords: list of coordinates, such that the nth entry corresponds
        to idx/ROI n in edges, above. It's fine if len(coords) > len(edges).
    :param arc_setting: changing between True/False will slightly change the
        look of the arcs.
    :param cbar: if not None, this argument will be called to make the colorbar.
        Otherwise, if None and not all the edge weights are equal to 1,
        then a colorbar will be created based on cmap and the edge_weights.
        If None and all the edge weights are equal to 1, then no colorbar
        will be created.
    :param do_ROI_circles: If True, then a small circle will be added for
        each ROI. If False, then this will not be done.
    :param do_ROI_circles_specific: If True, then small ROI circles will only
        be added for ROIs in edges used for arcs.
    :param ROI_circle_radius: The radius used for the ROI circles
    :param do_monkeypatch: There is seemingly a bug in matplotlib that prevents
        cleanly rotating characters. The matplotlib team has been notified about
        this, and a potential fix has been suggested. Until that is incorporated
        into matplotlib, the fix is being "monkeypatched". See
        chord.plot_rim_label(...) and patch_RenderAgg.py.
    :param edge_threshold: This parameter acts the same as edge_threshold in
        nilearn.plotting.plot_connectome. Edges whose abs(weight) is under
        the threshold are omitted. edge_threshold can be a float or a string
        representing a percentile (e.g., "25"). The latter causes edges below
        the percentile to be omitted.
    """

    network_order, network_colors = \
        _network_order_colors_check(network_order, network_colors, idx_to_label)

    if edge_weights is None:
        edge_weights = [1] * len(edges)
    else:
        edge_weights, edges = _threshold_proc(edge_threshold, edge_weights,
                                              edges)


    if vmin is None:
        vmin = min(edge_weights)
    if vmax is None:
        vmax = max(edge_weights)

    if vmin == vmax:
        raise ValueError(f'vmin and vmax cannot be equal. '
                         f'Note: your inputs only provide {len(edges)} edges '
                         f'after thresholding')

    if cmap is None:
        if abs(vmin - vmax) < 1e-6:
            cmap = plt.get_cmap('Greys')
        else:
            cmap = plt.get_cmap('turbo')
    elif isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if coords is None:
        coords = np.random.random(size=(len(idx_to_label), 3))

    plt.figure(figsize=(15, 15))
    radius = 0.6
    network_low_high, network_counts, network_centers, network_starts_ends = \
        plot_rim_and_labels(idx_to_label, network_order, network_colors, radius,
                            black_BG=black_BG, label_fontsize=label_fontsize,
                            do_monkeypatch=do_monkeypatch)


    vmin, vmax = plot_arcs(edges, idx_to_label, network_low_high, network_counts,
                           edge_weights, network_centers, network_starts_ends,
                           radius, cmap, coords, linewidths=linewidths,
                           seven_point_arc=arc_setting, colors=colors,
                           alphas=alphas, vmin=vmin, vmax=vmax,
                           plot_count=plot_count,
                           plot_abs_sum=plot_abs_sum,
                           norm_thickness=norm_thickness)

    if do_ROI_circles:
        if do_ROI_circles_specific:
            exclusive_idx = list(set(node_idx for tup in edges
                                     for node_idx in tup))
        else:
            exclusive_idx = None
        plot_ROI_circles(coords, idx_to_label, network_low_high, network_counts,
                         network_colors, radius,
                         exclusive_idx=exclusive_idx,
                         ROI_circle_radius=ROI_circle_radius)

    if cbar is not None:
        cbar()
    else:
        if not all(weight == 1 for weight in edge_weights):
            cbar = plt.colorbar(
                mappable=matplotlib.cm.ScalarMappable(cmap=cmap,
                                          norm=matplotlib.colors.Normalize(
                                          vmin=vmin, vmax=vmax)),
                                fraction=0.04, pad=0.04, ax=plt.gca())
            cbar.ax.tick_params(labelsize=30)

    plt.axis('off')
    plt.tight_layout()
    if fp_chord is not None:
        plt.savefig(fp_chord, dpi=dpi)
        plt.clf()


def plot_rim_and_labels(idx_to_label: dict, network_order: list,
                        network_colors: dict, radius: Union[float, int],
                        rim_border: Union[float, int] = 1.0,
                        black_BG: bool = False,
                        label_fontsize: int = 60,
                        do_monkeypatch: bool = True) -> Tuple[dict, dict,
                                                              dict, dict]:
    """
    Plots the chord diagram rims and labels. Each rim is plotted separately

    :param idx_to_label: dict mapping each ROI index to its chord label
    :param network_order: list specifying the order of the labels (rims)
    :param network_colors: dict mapping each label to a matplotlib color
    :param radius: Number specifying the radius of the outer part
    :param rim_border: Specifies the degrees/2 between each rim
        (the amount of white spacing between rims)
    :param do_monkeypatch: See chord.plot_rim_label(...) and patch_RenderAgg.py.
    :return: variables useful for add_arcs
        network_low_high: dict specifying (starting degree, ending degree)
            for each rim, including whitespace
        network_counts: dict specifying the number of ROIs for each rim
        network_center: dict specifying the center degree for each rim
    """
    num_ROIs = len(idx_to_label)

    network_counts = defaultdict(lambda: 0)
    for network in idx_to_label.values():
        network_counts[network] += 1

    network_low_high = {}
    network_center = {}
    network_starts_ends = {}
    circle_consumed = 0
    for i, network in enumerate(network_order):
        cnt = network_counts[network]
        degree_st = circle_consumed / num_ROIs * 360
        degree_end = (circle_consumed + cnt) / num_ROIs * 360
        if degree_end - degree_st < rim_border:
            warnings.warn(f'There are very few ROIs for the {network} network. '
                          f'May produce weirdness')
            rim_border_ = 0
        else:
            rim_border_ = rim_border
        plot_rim(degree_st, degree_end, rim_border=rim_border_, radius=radius,
                 color=network_colors[network])
        plot_rim_label(degree_st, degree_end, network, rim_border=rim_border_,
                       radius=radius,
                       color=network_colors[network],
                       label_fontsize=label_fontsize,
                       do_monkeypatch=do_monkeypatch)
        network_low_high[network] = (
            degree_st + rim_border_, degree_end - rim_border_)
        network_center[network] = (circle_consumed + cnt * 0.5) / num_ROIs * 360
        network_starts_ends[network] = (degree_st, degree_end)
        circle_consumed += cnt

    if black_BG:
        ax = plt.gca()
        black_circle = plt.Circle((0, 0), radius=radius - .0225, color='k')
        ax.add_patch(black_circle)
    return network_low_high, network_counts, network_center, network_starts_ends


def get_character_degree_locations(text: str, degree_st: Union[float, int],
                                   degree_end: Union[float, int],
                                   rim_border: Union[float, int],
                                   font_kwargs: dict) -> Tuple[list, str]:
    """
    Gets a list specifying the degree (location) of each character within
        the text. Also returns text, which may be shortened, if the full
        text does not fit within the edge.

    :param text: the string written around the rim
    :param degree_st: starting degree of the rim (whitespace included)
    :param degree_end: ending degree of the rim (whitespace included)
    :param rim_border: float specifying the degrees/2 between each rim
        (the amount of white spacing between rims)
    :param font_kwargs: dictionary used to specify text settings
    :return:
        char_degs = degree position of each character in text[::-1]
        text, may have been shortened if does not fully fit on the rim
    """
    r = plt.gcf().canvas.get_renderer()
    ax = plt.gca()
    arc_len = 2 * np.pi * (degree_end - degree_st - rim_border * 2) / 360
    char_degs = [0]
    char_deg = 0
    for char_i in range(0, len(text) - 1):
        char0 = text[char_i]
        char1 = text[char_i + 1]
        char0_transparent = plt.text(0, 0, char0, rotation=0,
                                     rotation_mode='anchor', ha='center',
                                     alpha=0, **font_kwargs)
        bb0 = char0_transparent.get_window_extent(renderer=r).transformed(
            ax.transData.inverted())
        char1_transparent = plt.text(0, 0, char1, rotation=0,
                                     rotation_mode='anchor', ha='center',
                                     alpha=0, **font_kwargs)
        bb1 = char1_transparent.get_window_extent(renderer=r).transformed(
            ax.transData.inverted())

        width = bb0.width / 2 + bb1.width / 2
        width *= 1.1
        boost = 1
        if 'fontname' in font_kwargs and \
                font_kwargs['fontname'].lower() == 'monospace':
            # Given the shapes of some letters, they look a bit better when
            # pushed to be closer to the previous/following letter in the text.
            if char1 in ['D', 'P', 'C']:
                boost -= .4
            elif char0 in ['A']:
                boost -= .8
            elif char1 in ['A']:
                boost += .4
            # Somebody who understands kerning please help me.
            # I've invested substantial effort into trying to get this right,
            # and this is the best that I've got. It works fine for the Yeo
            # atlas labels and for a few other label sets I tested, but
            # there could be label sets out there where it doesn't look good.

        char_deg += width / arc_len * (degree_end - degree_st) + boost
        if char_deg > degree_end - degree_st - rim_border * 2:
            text = text[:len(char_degs) - 1] + '.'
            break
        char_degs.append(char_deg)

    M = sum(char_degs) / len(char_degs)
    boost_need = (degree_end + degree_st) / 2 - M
    char_degs = [deg + boost_need for deg in char_degs]
    if text[-1] == '.':  # a period only takes up a small fraction of the its
                         # available horizontal space, so we shift all the
                         # characters to the left a bit to help make sure
                         # things keep looking centered
        chardot_transparent = plt.text(0, 0, '.', rotation=0,
                                       rotation_mode='anchor', ha='center',
                                       alpha=0, **font_kwargs)
        bb_dot = chardot_transparent.get_window_extent(renderer=r).transformed(
            ax.transData.inverted())
        subtract = bb_dot.width / arc_len * (degree_end - degree_st) * .75
        char_degs = [deg - subtract for deg in char_degs]

    return char_degs, text


def plot_rim_label(degree_st: Union[float, int], degree_end: Union[float, int],
                   text: str,
                   rim_border: Union[float, int] = 1, radius: float = 0.55,
                   color: Union[str, tuple] = 'black',
                   label_fontsize: Union[float, int] = 60,
                   do_monkeypatch: bool = True):
    """
    Adds the label for its corresponding rim.

    :param degree_st: Starting degree of the rim (whitespace included)
    :param degree_end: Ending degree of the rim (whitespace included)
    :param text: The string written around the rim
    :param rim_border: Float specifying the degrees/2 between each rim
                       (i.e., the amount of white spacing between rims)
    :param radius: radius of the chord diagram
    :param color: label color
    :param label_fontsize: label fontsize
    :param do_monkeypatch: I think there is a bug in matplotlib, which prevents
        characters from being rotated properly. patch_RendererAgg.py contains
        code that attempts to fix this. Setting do_monkeypatch == True applies
        this fix, while this script is running (to be clear, running this
        code will not permanently change your matplotlib files)
    """
    if do_monkeypatch:
        from nichord import patch_RendererAgg
        patch_RendererAgg.do_monkey_patch()

    # monospace is the easiest font to work with
    # although future work can try playing around with
    # the code to get non-monospace fonts working
    # (non-monospace is difficult to position).
    font_kwargs = {'fontname': 'monospace', 'fontsize': label_fontsize}
    char_degrees, text = get_character_degree_locations(text, degree_st,
                                                        degree_end, rim_border,
                                                        font_kwargs)

    text = text[::-1]  # The text must be flipped so that the text reads
                 # left-to-right (note that the last character must have
                 # the lowest degree angle.
    # spacing_incrament
    for deg, char in zip(char_degrees, text):
        trans_deg = deg
        trans_deg270 = deg + 270
        x = np.cos(trans_deg / 360 * 2 * np.pi) * (radius)
        y = np.sin(trans_deg / 360 * 2 * np.pi) * (radius)
        plt.text(x, y, char, rotation=trans_deg270, rotation_mode='anchor',
                 va='bottom',
                 ha='center', color=color, **font_kwargs)  # note that


def plot_rim(degree_st: Union[float, int], degree_end: Union[float, int],
             rim_border: Union[float, int] = 1,
             radius: float = 0.55, color: Union[str, tuple] = 'black'):
    """
    The outside rims of the chord diagram are plotted as one colored Wedge
        underneath a smaller white Wedge.

    :param degree_st: starting degree of the rim (whitespace included)
    :param degree_end: ending degree of the rim (whitespace included)
    :param rim_border: Float specifying the degrees/2 between each rim
                       (i.e., the amount of white spacing between rims)
    :param radius: radius of the chord diagram
    :param color: rim color
    """
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.xlim(-0.65, 0.65)
    plt.ylim(-0.65, 0.65)
    ax.add_patch(patches.Wedge((.0, .0), radius, degree_st + rim_border,
                               degree_end - rim_border, color=color,
                               alpha=1))  # larger, colored wedge
    ax.add_patch(patches.Wedge((.0, .0), radius - .02, degree_st, degree_end,
                               color='white'))  # smaller white edge


def plot_arcs(edges: list, idx_to_label: dict, network_low_high: dict,
              network_counts: dict,
              edge_weights: list,
              network_centers: dict,
              network_starts_ends: dict,
              radius: float,
              cmap: Union[None, str, matplotlib.colors.Colormap],
              coords: list,
              colors: Union[None, str, tuple, list],
              linewidths: Union[None, float, int, list],
              alphas: Union[None, float, int, list],
              seven_point_arc: bool = False,
              vmin: Union[float, int] = -1,
              vmax: Union[float, int] = 1,
              plot_count: bool = True,
              plot_abs_sum: bool = False,
              norm_thickness: bool = False,
              max_linewidth: Union[float, int] = 28,
              sub_min_thickness: bool = False,
              ) -> (float, float):
    """
    Plots the arcs between each ROI. Within the rim/label, ROIs are positioned
        based on their y coordinate (i.e., how anterior/posterior it is). If
        no coordinate is provided, the ROIs are positioned randomly
        (see plot_chord_diagram(...)).

    :param edges: list of tuples (a, b), where a = index of the 1st ROI &
        b = index of the 2nd ROI
    :param idx_to_label: dict mapping each ROI index to its chord label
    :param network_low_high: See plot_rim(...)
    :param network_counts: See plot_rim(...)
    :param edge_weights: list of edge weights. The nth weight, here, should
        correspond to the nth edge in edges, above
    :param network_centers: See plot_rim(...)
    :param radius: radius of the outer part of the rim/circle
    :param cmap: can either be None (default will be used), a string (will
        retrieve cmap from matplotlib), or a matplotlib colormap
    :param coords: list of coordinates, such that the nth entry corresponds to
        idx/ROI n in edges, above
    :param colors: List of colors for the edges. If None, then
        some will be generated based on the magnitudes of the weights.
    :param linewidths: Same as colors.
    :param alphas: Same as colors.
    :param seven_point_arc: Changing between 0 or 1 will slightly change the
        look of the arcs
    """
    roi2degree = ROI_to_degree(coords, idx_to_label, network_low_high,
                               network_counts)
    if edge_weights is None: edge_weights = np.random.random(len(edges)) * 2 - 1

    if plot_count or plot_abs_sum:
        network_edges = OrderedDict()
        for (i, j), w in zip(edges, edge_weights):
            if i == j: continue # sometimes edges connected to themselves
                                # are added. These should be ignored
            i_network = idx_to_label[i]
            j_network = idx_to_label[j]
            if i_network >= j_network:
                if (i_network, j_network) not in network_edges:
                    network_edges[(i_network, j_network)] = []
                network_edges[(i_network, j_network)].append(w)
            elif j_network > i_network:
                if (j_network, i_network) not in network_edges:
                    network_edges[(j_network, i_network)] = []
                network_edges[(j_network, i_network)].append(w)
        if plot_abs_sum:
            network2thickness = {network: sum(np.abs(l_w))
                                 for network, l_w
                                 in network_edges.items()}
        else:
            network2thickness = {network: len(l_w)
                                 for network, l_w
                                 in network_edges.items()}
        if norm_thickness:
            network2thickness_std = {}
            min_std_thickness = 1e10
            max_std_thickness = 0
            for network_pair, thickness in network2thickness.items():
                network_span0 = network_starts_ends[network_pair[0]][1] - \
                                network_starts_ends[network_pair[0]][0]
                network_span1 = network_starts_ends[network_pair[1]][1] - \
                                network_starts_ends[network_pair[1]][0]
                if network_pair[0] == network_pair[1]:
                    expected_thickness = (network_span0*network_span1) / (720*360)
                else:
                    expected_thickness = (network_span0*network_span1) / (360*360)
                network2thickness_std[network_pair] = \
                    thickness / expected_thickness * 25
                min_std_thickness = min(min_std_thickness,
                                        network2thickness_std[network_pair])
                max_std_thickness = max(max_std_thickness,
                                        network2thickness_std[network_pair])
            if True or sub_min_thickness:
                min_std_thickness -= .1
                network2thickness_std = {network_pair:
                                         (thickness - min_std_thickness) /
                                         (max_std_thickness - min_std_thickness)
                                     for network_pair, thickness in
                                     network2thickness_std.items()}
            network2thickness = network2thickness_std

        total_thickness = sum(network2thickness.values()) # diff. normalization
        max_linewidth_seen = 0
        for edge, thickness in network2thickness.items(): # ensures never too thick
            linewidth = thickness / total_thickness * 200
            max_linewidth_seen = max(max_linewidth_seen, linewidth)
        if max_linewidth_seen > max_linewidth:
            for edge, thickness in network2thickness.items():
                network2thickness[edge] = thickness / max_linewidth_seen * \
                                          max_linewidth
        edge_weights = [np.mean(l_w) for l_w in network_edges.values()]
        vmin = min(edge_weights)
        vmax = max(edge_weights)
        edges = list(network_edges.keys())
        seven_point_arc = False

    if plot_count or plot_abs_sum:
        edges, edge_weights = zip(*sorted(zip(edges, edge_weights),
                                      key=lambda x: network2thickness[x[0]]))
    else:
        edges, edge_weights = zip(*sorted(zip(edges, edge_weights),
                                      key=lambda x: abs(x[1])))
    for idx, (edge, edge_weight) in enumerate(zip(edges, edge_weights)):
        if plot_count or plot_abs_sum:
            deg0 = network_centers[edge[0]]
            deg1 = network_centers[edge[1]]
            network_span0 = network_starts_ends[edge[0]][1] - \
                            network_starts_ends[edge[0]][0]
            network_span1 = network_starts_ends[edge[1]][1] - \
                            network_starts_ends[edge[1]][0]

            if edge[0] == edge[1]:
                deg0 -= 0.15 * network_span0
                deg1 += 0.15 * network_span1
            else:
                deg0_ = deg0
                deg1_ = deg1
                if abs(deg0 + 360 - deg1) < abs(deg0 - deg1):
                    deg1_ -= 360
                if abs(deg1 + 360 - deg0) < abs(deg0 - deg1):
                    deg0_ -= 360

                assert abs(deg0_ - deg1_) <= 180, f'Bad: {deg0_} | {deg1_}'
                rel_dist = (deg0_ - deg1_) / 180
                if rel_dist < 0:
                    rel_dist_ = -1 - rel_dist
                elif rel_dist > 0:
                    rel_dist_ = 1 - rel_dist
                rel_dist_ = np.sign(rel_dist_) * (abs(rel_dist_) ** 2)
                deg0 -= rel_dist_ * network_span0/2
                deg1 += rel_dist_ * network_span1/2

        else:
            deg0 = roi2degree.get_degree(edge[0])
            deg1 = roi2degree.get_degree(edge[1])
        color = None
        linewidth = None
        alpha = None
        if all(weight == 1 for weight in edge_weights):
            if colors is None:
                color = 'k'
            if linewidths is None:
                linewidth = 1.7
            if alphas is None:
                alpha = 0.5
        else:
            if colors is None:
                weight_relative_min = (edge_weight - vmin) / \
                                      (vmax - vmin)
                color = cmap(weight_relative_min)
            if linewidths is None:
                if plot_count or plot_abs_sum:
                    linewidth = network2thickness[edge] / total_thickness * 250
                else:
                    linewidth = 1.7 + abs(edge_weight)
            if alphas is None:
                alpha = 0.7

        if color is None:
            if isinstance(colors, list):
                color = colors[idx]
            else:
                color = colors

        if linewidth is None:
            if isinstance(linewidths, list):
                linewidth = linewidths[idx]
            else:
                linewidth = linewidths

        if alpha is None:
            if isinstance(alphas, list):
                alpha = alphas[idx]
            else:
                alpha = alphas

        if abs(deg0 - deg1) < .000000000001:
            continue     # ignores nodes' connections to itself

        if plot_count or plot_abs_sum:
            center0 = network_centers[edge[0]]
            center1 = network_centers[edge[1]]
        else:
            center0 = network_centers[idx_to_label[edge[0]]]
            center1 = network_centers[idx_to_label[edge[1]]]

        plot_arc(deg0, deg1,
                 center0, center1,
                 radius, color=color,
                 linewidth=linewidth, alpha=alpha,
                 seven_point_arc=seven_point_arc)
    return vmin, vmax


def plot_arc(deg0: Union[int, float], deg1: [int, float],
             rim_center_deg0: [int, float], rim_center_deg1: [int, float],
             radius: float, alpha: float = 0.5,
             color: Union[str, tuple] = 'black', seven_point_arc: bool = False,
             linewidth: Union[float, int] = 1):
    """
    Adds the arc between the two ROIs. The arcs are either made up of 5 or 7
        connected and smoothed points.

    :param deg0: degree position of ROI 0
    :param deg1: degree position of ROI 1
    :param rim_center_deg0: degree of the center of the rim corresponding to
        ROI 0 (this helps with making nice arcs)
    :param rim_center_deg1: degree of the center of the rim corresponding to
        ROI 1 (this helps with making nice arcs)
    :param radius: Radius of the outer part of the rim/circle
    :param alpha: corresponds to the arcs
    :param color: Color of the arcs
    :param seven_point_arc: Tweaking True/False changes the look of the arcs
    :param linewidth: Width of arcs
    """


    theta0 = radians(deg0)
    theta1 = radians(deg1)
    rim_center_theta0 = radians(rim_center_deg0)
    rim_center_theta1 = radians(rim_center_deg1)

    radius = radius - 0.02
    start_cart = polar_to_cart(radius, theta0)
    end_cart = polar_to_cart(radius, theta1)

    if abs(deg0 - deg1) < .000001:
        # Code goes here...
        return

    theta0_ = (theta0 + np.pi) % (2 * np.pi) - np.pi
    theta1_ = (theta1 + np.pi) % (2 * np.pi) - np.pi
    dif = abs(theta0_ - theta1_)
    dif = 2 * np.pi - dif if dif > np.pi else dif

    start_core = [radius * (0.5 + 0.5 * (1 - (dif / np.pi)) ** 0.5),
                  (theta0 + rim_center_theta0) / 2]
    end_core = [radius * (0.5 + 0.5 * (1 - (dif / np.pi)) ** 0.5),
                (theta1 + rim_center_theta1) / 2]

    start_network_cart = polar_to_cart(*start_core)

    end_network_cart = polar_to_cart(*end_core)


    start_inner_cart = polar_to_cart(radius - .02, theta0)
    end_inner_cart = polar_to_cart(radius - .02, theta1)

    theta_mid = (theta0 + theta1) / 2
    if abs(theta1 - theta0) > np.pi:
        theta_mid += np.pi




    less_than_half_away = abs(theta0 - theta1) < np.pi / 2 or \
                          abs(theta0 - theta1 - 2 * np.pi) < np.pi / 2 or \
                          abs(theta0 - theta1 + 2 * np.pi) < np.pi / 2

    # same_network = abs(rim_center_deg0 - rim_center_deg1) < .0001
    # less_than_half_away
    if not seven_point_arc or less_than_half_away:
        dif_radius = (radius - .02) * ((1 - (dif / np.pi)) ** 1.35)
        mid_cart = polar_to_cart(dif_radius, theta_mid)
        x, y = zip(start_cart, start_inner_cart, mid_cart,
                   end_inner_cart, end_cart)
        warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
        tck, u = interpolate.splprep([list(x), list(y)], k=3,
                                     s=.0002,
                                     quiet=True)
        u = np.linspace(0, 1, 1000)
        xnew, ynew = interpolate.splev(u, tck)
        xnew, ynew = zip(
            *filter(lambda xy: cart_to_polar(*xy)[0] < radius,
                    zip(xnew, ynew)))
    else:
        dif_radius = radius * ((1 - (dif / np.pi)) ** 1.5)
        mid_cart = polar_to_cart(dif_radius, theta_mid)
        x, y = zip(start_cart, start_inner_cart, start_network_cart, mid_cart,
                   end_network_cart, end_inner_cart, end_cart)
        tck, u = interpolate.splprep([list(x), list(y)], k=3, s=0.0002,
                                     quiet=True)
        u = np.linspace(0, 1, 1000)
        xnew, ynew = interpolate.splev(u, tck)
        xnew, ynew = zip(
            *filter(lambda xy: cart_to_polar(*xy)[0] < radius,
                    zip(xnew, ynew)))

    plt.gca().plot(xnew, ynew, alpha=alpha, linewidth=linewidth, color=color,
                   solid_joinstyle='bevel', solid_capstyle='round')


def plot_ROI_circles(coords: list, idx_to_label: dict, network_low_high: dict,
                     network_counts: dict,
                     network_colors: dict, radius: float,
                     exclusive_idx: Union[None, list, set]=None,
                     ROI_circle_radius=0.005) -> None:
    """
    Optional piece of the chord diagram. This will draw a little circle at each
        ROI's position within each rim.

    :param coords: list of coordinates, such that the nth entry corresponds to
        idx/ROI n in edges (see other functions)
    :param idx_to_label: dict mapping each ROI index to its chord label
        (e.g., {0: "FPCN"})
    :param network_low_high: see plot_rims(...)
    :param network_counts: see plot_rims(...)
    :param network_colors: dict mapping each label to a matplotlib color
    :param radius: float specifying the radius of the outer part
    """
    ax = plt.gca()
    roi2degree = ROI_to_degree(coords, idx_to_label, network_low_high,
                               network_counts)
    for idx in idx_to_label:
        if exclusive_idx is not None and idx not in exclusive_idx:
            continue
        deg = roi2degree.get_degree(idx)
        x, y = polar_to_cart(radius - 0.02, radians(deg))
        circle = plt.Circle((x, y), ROI_circle_radius,
                            color=network_colors[idx_to_label[idx]],
                            zorder=100000)
        ax.add_patch(circle)


def polar_to_cart(r: Union[float, int], theta: Union[float, int]) -> Tuple[
    float, float]:
    """
    Converts polar points (r, theta) to cartesian (x, y)
    """
    return r * math.cos(theta), r * math.sin(theta)


def cart_to_polar(x: Union[float, int], y: Union[float, int]) -> Tuple[
    float, float]:
    """
    Converts cartesian points (x, y) to polar (r, theta)
    """
    z = x + y * 1j
    r, theta = np.abs(z), np.angle(z)
    return r, theta


def _threshold_proc(edge_threshold, edge_weights, edges):
    if edge_threshold == 0: return edge_weights, edges
    edge_threshold = check_threshold(
        edge_threshold,
        np.abs(edge_weights),
        scoreatpercentile,
        "edge_threshold",
    )

    if edge_threshold > 0:
        edges = [edge for edge, weight in zip(edges, edge_weights)
                 if abs(weight) > edge_threshold]
        edge_weights = [weight for weight in edge_weights
                        if abs(weight) > edge_threshold]
    return edge_weights, edges
