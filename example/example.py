from nichord.chord import plot_chord
from nichord.glassbrain import plot_glassbrain
from nichord.combine import combine_imgs, plot_and_combine
from nichord.coord_labeler import get_idx_to_label


# Example 0 described in README.md
if __name__ == '__main__':
    edges = [(0, 1), (0, 2), (1, 5), (3, 5), (4, 6), (2, 7), (6, 7)]
    edge_weights = [-0.3, -0.5, 0.7, 0.5, -0.2, 0.3, 0.8]
    coords = [[-24, -99, -12], [51, -3, -15], [-15, -70, 30], [21, 39, 39],
              [21, -66, 48], [54, 33, 12], [-33, 3, 3], [57, -45, 12]]
    idx_to_label = get_idx_to_label(coords, atlas='yeo')

    fp_chord = r'example/chord/ex0_chord.png'
    plot_chord(idx_to_label, edges, edge_weights=edge_weights,
               coords=None, network_order=None,
               network_colors=None, linewidths=15,
               alphas = 0.9,
               fp_chord=fp_chord, do_ROI_circles=True,
               do_ROI_circles_specific=True, ROI_circle_radius=0.02,
               arc_setting=False)

    fp_glass = r'example/glass/ex0_glass.png'
    plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                    coords, linewidths=15, node_size=17)

    fp_combined = r'example/ex0.png'
    combine_imgs(fp_glass, fp_chord, fp_combined)

# Beginning of examples 1 and 2. These examples reflect how NiChord may be used
#   in a project involving a full connectome's worth of ROIs. Here ROIs are
#   taken from the Power et al. (2014) atlas. The examples also show how
#   network label colors can be specified by the user and the networks can be
#   plotted in a specific order around the circle.
# It also shows how to plot everything at once using combine.plot_and_combine

if __name__ == '__main__':
    # example 1 ---------------------

    network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                      'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                      'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

    network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic',
                     'Uncertain', 'VAN']

    power_coords = [[-24, -99, -12], [27, -96, -12], [24, 33, -18], [-57, -45, -24], [9, 42, -24], [-21, -21, -21],
                    [18, -27, -18], [-36, -30, -27], [66, -24, -18], [51, -33, -27], [54, -30, -18], [33, 39, -12],
                    [-6, -51, 60], [-15, -18, 39], [0, -15, 48], [9, -3, 45], [-6, -21, 66], [-6, -33, 72],
                    [12, -33, 75], [-54, -24, 42], [30, -18, 72], [9, -45, 72], [-24, -30, 72], [-39, -18, 54],
                    [30, -39, 60], [51, -21, 42], [-39, -27, 69], [21, -30, 60], [45, -9, 57], [-30, -42, 60],
                    [9, -18, 75], [21, -42, 69], [-45, -33, 48], [-21, -30, 60], [-12, -18, 75], [42, -21, 54],
                    [-39, -15, 69], [-15, -45, 72], [3, -27, 60], [3, -18, 57], [39, -18, 45], [-48, -12, 36],
                    [36, -9, 15], [51, -6, 33], [-54, -9, 24], [66, -9, 24], [-3, 3, 54], [54, -27, 33], [18, -9, 63],
                    [-15, -6, 72], [-9, -3, 42], [36, 0, -3], [12, 0, 69], [6, 9, 51], [-45, 0, 9], [48, 9, 0],
                    [-33, 3, 3], [-51, 9, -3], [-6, 18, 33], [36, 9, 0], [33, -27, 12], [66, -33, 21], [57, -15, 6],
                    [-39, -33, 18], [-60, -24, 15], [-48, -27, 6], [42, -24, 21], [-51, -33, 27], [-54, -21, 24],
                    [-54, -9, 12], [57, -6, 12], [60, -18, 30], [-30, -27, 12], [-42, -75, 27], [6, 66, -3],
                    [9, 48, -15], [-12, -39, 0], [-18, 63, -9], [-45, -60, 21], [42, -72, 27], [-45, 12, -33],
                    [45, 15, -30], [-69, -24, -15], [-57, -27, -15], [27, 15, -18], [-45, -66, 36], [-39, -75, 45],
                    [-6, -54, 27], [6, -60, 36], [-12, -57, 15], [-3, -48, 12], [9, -48, 30], [15, -63, 27],
                    [-3, -36, 45], [12, -54, 18], [51, -60, 36], [24, 33, 48], [-9, 39, 51], [-15, 30, 54],
                    [-36, 21, 51], [21, 39, 39], [12, 54, 39], [-9, 54, 39], [-21, 45, 39], [6, 54, 15], [6, 63, 21],
                    [-6, 51, 0], [9, 54, 3], [-3, 45, -9], [9, 42, -6], [-12, 45, 9], [-3, 39, 36], [-3, 42, 15],
                    [-21, 63, 18], [-9, 48, 24], [66, -12, -18], [-57, -12, -9], [-57, -30, -3], [66, -30, -9],
                    [-69, -42, -6], [12, 30, 60], [12, 36, 21], [51, -3, -15],
                    [-26.833333333333332, -39.0, -8.833333333333334], [27, -36, -12], [-33, -39, -15], [27, -78, -33],
                    [51, 6, -30], [-54, 3, -27], [48, -51, 30], [-48, -42, 0], [-30, 18, -18], [-3, -36, 30],
                    [-6, -72, 42], [12, -66, 42], [3, -48, 51], [-45, 30, -12], [-9, 12, 66], [48, 36, -12],
                    [9, -90, -6], [18, -90, -15], [-12, -96, -12], [18, -48, -9], [39, -72, 15], [9, -72, 12],
                    [-9, -81, 6], [-27, -78, 18], [21, -66, 3], [-24, -90, 18], [27, -60, -9], [-15, -72, -9],
                    [-18, -69, 6], [42, -78, -12], [-48, -75, -9], [-15, -90, 30], [15, -87, 36], [30, -78, 24],
                    [21, -87, -3], [15, -78, 30], [-15, -51, 0], [42, -66, -9], [24, -87, 24], [6, -72, 24],
                    [-42, -75, 0], [27, -78, -15], [-15, -78, 33], [-3, -81, 21], [-39, -87, -6], [36, -84, 12],
                    [6, -81, 6], [-27, -90, 3], [-33, -78, -12], [36, -81, 0], [-45, 3, 45], [48, 24, 27],
                    [-48, 12, 24], [-54, -48, 42], [-24, 12, 63], [57, -54, -15], [24, 45, -15], [33, 54, -12],
                    [-21, 42, -21], [-18, -75, -24], [18, -81, -33], [36, -66, -33], [48, 9, 33], [-42, 6, 33],
                    [-42, 39, 21], [39, 42, 15], [48, -42, 45], [-27, -57, 48], [45, -54, 48], [33, 15, 57],
                    [36, -66, 39], [-42, -54, 45], [39, 18, 39], [-33, 54, 3], [-42, 45, -3], [33, -54, 45],
                    [42, 48, -3], [-42, 24, 30], [-3, 27, 45], [12, -39, 51], [54, -45, 36], [42, 0, 48], [30, 33, 27],
                    [48, 21, 9], [-36, 21, 0], [36, 21, 3], [36, 33, -3], [33, 15, -9], [-12, 27, 24], [0, 15, 45],
                    [-27, 51, 21], [0, 30, 27], [6, 24, 36], [9, 21, 27], [30, 57, 15], [27, 51, 27], [-39, 51, 18],
                    [3, -24, 30], [6, -24, 0], [-3, -12, 12], [-9, -18, 6], [12, -18, 9], [-6, -27, -3], [-21, 6, -6],
                    [-15, 3, 9], [30, -15, 3], [24, 9, 0], [29.833333333333332, 0.0, 3.1666666666666665], [-30, -12, 0],
                    [15, 6, 6], [9, -3, 6], [54, -42, 21], [-57, -51, 9], [-54, -39, 15], [51, -33, 9], [51, -30, -3],
                    [57, -45, 12], [54, 33, 0], [-48, 24, 0], [-15, -66, -21], [-33, -54, -24], [21, -57, -24],
                    [0, -63, -18], [33, -12, -33], [-30, -9, -36], [48, -3, -39], [-51, -6, -39], [9, -63, 60],
                    [-51, -63, 6], [-48, -51, -21], [45, -48, -18], [48, -30, 48], [21, -66, 48], [45, -60, 3],
                    [24, -57, 60], [-33, -45, 48], [-27, -72, 36], [-33, 0, 54], [-42, -60, -9], [-18, -60, 63],
                    [30, -6, 54]]

    edges = [(190, 204), (198, 204), (198, 190), (212, 204), (212, 190), (212, 198), (164, 204), (164, 190), (164, 198),
             (164, 212), (174, 204), (174, 190), (174, 198), (174, 212), (174, 164), (168, 204), (168, 190), (168, 198),
             (168, 212), (168, 164), (168, 174), (259, 204), (259, 190), (259, 198), (259, 212), (259, 164), (259, 174),
             (259, 168), (185, 204), (185, 190), (185, 198), (185, 212), (185, 164), (185, 174), (185, 168), (185, 259),
             (260, 204), (260, 190), (260, 198), (260, 212), (260, 164), (260, 174), (260, 168), (260, 259), (260, 185),
             (146, 204), (146, 190), (146, 198), (146, 212), (146, 164), (146, 174), (146, 168), (146, 259), (146, 185),
             (146, 260), (145, 204), (145, 190), (145, 198), (145, 212), (145, 164), (145, 174), (145, 168), (145, 259),
             (145, 185), (145, 260), (145, 146), (189, 204), (189, 190), (189, 198), (189, 212), (189, 164), (189, 174),
             (189, 168), (189, 259), (189, 185), (189, 260), (189, 146), (189, 145), (186, 204), (186, 190), (186, 198),
             (186, 212), (186, 164), (186, 174), (186, 168), (186, 259), (186, 185), (186, 260), (186, 146), (186, 145),
             (186, 189), (139, 204), (139, 190), (139, 198), (139, 212), (139, 164), (139, 174), (139, 168), (139, 259),
             (139, 185), (139, 260), (139, 146), (139, 145), (139, 189), (139, 186), (153, 204), (153, 190), (153, 198),
             (153, 212), (153, 164), (153, 174), (153, 168), (153, 259), (153, 185), (153, 260), (153, 146), (153, 145),
             (153, 189), (153, 186), (153, 139)]

    edge_weights = [-0.10596976331347109, -0.11813383536048842, 0.027596255165194213, -0.031922521544466585,
                    -0.11174678377038612, 0.28993386480373207, 0.20054896659520055, 0.2553259315088739,
                    0.07653428926238003, -0.7055275397448448, -0.034075825047334356, 0.2813393925738696,
                    0.054414395898237954, -0.06203263374523441, 0.15872492235915645, 0.20604616340275936,
                    0.23635602352560806, 0.0022691567868356414, -0.053534754420220654, 0.3920469304141896,
                    0.4075702961316909, 0.46027276552407764, 0.4227594033198163, 0.35510617227489516,
                    -0.22775858350448316, 0.28739315085654266, 0.03751279250284278, 0.28791751633224516,
                    0.3246660479804881, 0.07112569082364452, 0.0924702524366891, 0.23884561722099315,
                    0.14021272340580332, -0.10430273964061443, -0.21265179164804557, -0.03609932447351462,
                    0.14605848267238317, -0.02745371747098601, 0.27561248906826497, 0.37943103912594767,
                    0.13021326861700808, -0.208752476761443, -0.14671188616188346, -0.22188640808442398,
                    0.1238249017211098, 0.2158897849274389, -0.015531387344334851, 0.03266414050904938,
                    0.05420454970358559, 0.052179453146439146, -0.4847014921258607, 0.4409560598226891,
                    0.15876087584499823, -0.12072561319083305, 0.02951073269677679, 0.14788563998136903,
                    -0.007302764669301237, -0.0831049201842025, -0.2650790942589829, 0.043604680854305584,
                    0.2167352587914131, 0.18689110656395508, 0.379790340539003, 0.26979325989595104,
                    0.29352296948542406, 0.6346407634533036, -0.06013758154985464, 0.3996252203382274,
                    0.3199358472225261, 0.36362330077730387, -0.37040814121339644, 0.03757112472946471,
                    0.21143280680947202, -0.3397290023352397, 0.44927701334449194, 0.29328295901562573,
                    0.0752617752232051, -0.02383818190465763, 0.7989809120967143, 1.0, 0.05703954827231009,
                    0.40512038299932124, -0.21480087124077574, 0.5243127651712001, 0.132801413714605,
                    0.4367804489407866, 0.3210727264087059, 0.00842978235176351, -0.08531289252876859,
                    0.2869625635474946, 0.36548825521776573, 0.5292547417718756, -0.22209683530448307,
                    0.25508627265856704, 0.20233858889162704, -0.11385190931824272, -0.10393565691252553,
                    -0.1969561280044689, 0.250875554996767, 0.03001678499212991, -0.0867174433985537,
                    0.3052894863215727, 0.0733640229497709, 0.11925270709307856, 0.06035105949695806,
                    0.21371750003814022, 0.27166320103818403, 0.04187229536399407, -0.1623916304029322,
                    0.07636769790344225, 0.0783496321702299, 0.3171764690748661, 0.48264021671410895,
                    -0.15291727534860514, -0.217070036254596, 0.2516696298803447, -0.07040341280408653,
                    0.1544046218403957, 0.017626751502194702, -0.35345097176546175]


    idx_to_label = get_idx_to_label(power_coords, atlas='yeo')

    dir_out = 'example'
    fn = 'ex1.png'
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=power_coords,
                     network_order=network_order, network_colors=network_colors,
                     )

    fn = r'ex1_black_BG.png'
    chord_kwargs = {'black_BG': True}
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=power_coords,
                     network_order=network_order, network_colors=network_colors,
                     chord_kwargs=chord_kwargs, title='Example 1b (black)')

    fn = r'ex1_count.png'
    chord_kwargs = {'plot_count': True}
    n_nodes = len(set([i for i, j in edges] + [j for i, j in edges]))
    glass_kwargs = {'linewidths': 0.,
                    'node_size': list(range(1, n_nodes+1))}
    # linewidths = 0 for glass_kwargs causes no lines to be plotted on brain
    # setting a list for node_sizes causes the nodes to be plotted in dif sizes
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=power_coords,
                     network_order=network_order, network_colors=network_colors,
                     chord_kwargs=chord_kwargs, glass_kwargs=glass_kwargs,
                     title='Example 1c (count)')


    # example 2 ---------------------
    fn = r'ex2.png'
    edges = [(32, 12), (32, 48), (33, 48), (101, 105), (105, 219), (201, 33),
             (32, 105)]
    edge_weights = [-0.3, -0.5, 0.7, 0.5, -0.2, 0.3, 0.8]

    chord_kwargs = {'alphas': 0.9, 'linewidths': 15, 'do_ROI_circles': True,
                    'do_ROI_circles_specific': True, 'ROI_circle_radius': 0.02,
                    'arc_setting': False}
    glass_kwargs = {'linewidths': 15, 'node_size': 17}
    plot_and_combine(dir_out, fn, idx_to_label, edges,
                     edge_weights=edge_weights, coords=power_coords,
                     network_order=network_order, network_colors=network_colors,
                     chord_kwargs=chord_kwargs, glass_kwargs=glass_kwargs,
                     title='Example 2')