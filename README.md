![PyPI](https://img.shields.io/pypi/v/nichord) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/nichord?link=https%3A%2F%2Fpypi.org%2Fproject%2Fnichord%2F)


# NiChord

`NiChord` is a Python package for visualizing functional connectivity data. This package was inspired by [NeuroMArVL](https://immersive.erc.monash.edu/neuromarvl/?example=40496078-effa-4ac3-9d3e-cb7f946e7dd1_137.147.133.145), an online visualization tool. Although the code was designed for neuroscience research, it can be used with any configuration of edge and label data.

<p align="center">
  <img src="example\outside_chord_example.png"  width="600" />
</p>

<p align="center">
  <img src="example\ex1.png"  width="600" />
</p>

## Installation
Can be installed via pip:

```
pip install nichord
```

Can alternatively be installed via conda:

```
conda install -c conda-forge nichord
```

## Examples

Here, we cover the code provided in `example\example.py`. 

### Input variables
`edges` are specified as a list of tuples, `(i, j)`, where `i` and `j` are indices representing the two nodes making up the edge. For this example, the following list represents seven edges among eight nodes.

```Python
edges = [(0, 1), (0, 2), (1, 5), (3, 5), (4, 6), (2, 7), (6, 7)]
```

Each node index should also correspond to a coordinate in MNI space. The coordinates are defined as a list of lists or list of tuples, wherein the outer list's length (8, here) corresponds to the number of nodes. 
```Python
coords = [[-24, -99, -12], [51, -3, -15], [-15, -70, 30], [21, 39, 39],
          [21, -66, 48], [54, 33, 12], [-33, 3, 3], [57, -45, 12]]
```

These coordinates can be used to construct a dictionary (`idx_to_label`), mapping each node index to a network label. This package provides functions to help assign labels to nodes given their anatomical location, based on the [Yeo et al. (2011) atlas](https://journals.physiology.org/doi/full/10.1152/jn.00338.2011):

```Python
from nichord.coord_labeler import get_idx_to_label
idx_to_label = get_idx_to_label(coords, atlas='yeo')
```

`idx_to_label` can alternatively be defined manually:

```Python
idx_to_label = {0: 'Visual', 1: 'DMN', 2: 'Visual', 3: 'DMN', 
                4: 'DAN', 5: 'FPCN', 6: 'VAN', 7: 'VAN'}
```

You may assign each edge a weight. Weights are defined as a list of length equal to the number of edges (e.g., 7 weights for this example). If `edge_weights = None`, then grey edges are plotted, unless an aggregation feature is used (see `plot_count=True` below).

```Python
edge_weights = [-0.3, -0.5, 0.7, 0.5, -0.2, 0.3, 0.8]
```

### Plotting

These variables and an optional filepath can then be passed to create a chord diagram:

```Python
from nichord.chord import plot_chord

# If the filepath is left None, the chord diagram can be opened in a matplotlib with plt.show()
fp_chord = 'ex0_chord.png'
plot_chord(idx_to_label, edges, edge_weights=edge_weights, fp_chord=fp_chord, 
           linewidths=15, alphas=0.9, do_ROI_circles=True, label_fontsize=70, 
           # July 2023 update allows changing label fontsize
           do_ROI_circles_specific=True, ROI_circle_radius=0.02)

```

<p align="center">
  <img src="example\chord\ex0_chord.png" width="400" >
  <br>
  If no filepath is passed, the diagram will be opened in a matplotlib window.
</p>


The code can also be used to plot glass brains, which leverages the same variables. Note that the colors of the glass brain nodes correspond to the network colors in the chord diagram.

```Python
from nichord.glassbrain import plot_glassbrain

fp_glass = 'ex0_glassbrain.png'
plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                coords, linewidths=15, node_size=17)
```

<p align="center">
  <img src="example\glass\ex0_glass.png" width="760"/>
</p>

You can combine the figures above into a single figure.

```Python
from nichord.combine import combine_imgs

fp_combined = 'ex0_combined.png'
combine_imgs(fp_glass, fp_chord, fp_combined)
```

Notably, these functions have many other optional variables (e.g., passing specific colors for each network label using a dictionary). 

<p align="center">
  <img src="example\ex0.png" width="800"/>
</p>

Further information on these optional variables can be seen by examining `example\example.py`, which contains code used to generate the two examples from the first section of this README. You can also learn about these by reading the hinting/documentation within each function.

### Plotting everything at once

You can also use `combine.plot_and_combine` to do `plot_chord`, `plot_glassbrain`, and `combine_image` with a single function. `plot_and_combine` will create (if needed) and use directories `chord` and `glass` wherever you specify the combined image to be made with `dir_out`.

```Python

network_colors = {'Uncertain': 'black', 'Visual': 'purple',
                  'SM': 'darkturquoise', 'DAN': 'green', 'VAN': 'fuchsia',
                  'Limbic': 'burlywood', 'FPCN': 'orange', 'DMN': 'red'}

network_order = ['FPCN', 'DMN', 'DAN', 'Visual', 'SM', 'Limbic', 
                 'Uncertain', 'VAN']

dir_out = 'example'
fn = 'ex1.png'
plot_and_combine(dir_out, fn, idx_to_label, edges,
                 edge_weights=edge_weights, coords=power_coords,
                 network_order=network_order, network_colors=network_colors)
```

To `plot_and_combine`, you can pass `chord_kwargs` and/or `glass_kwargs` to adjust the appearance of the chord diagram or glass brain, which will in turn be sent to  `plot_chord` and/or `plot_glassbrain`. The example below shows this and also how you can add a title and give the chord diagram a black background:

```Python
dir_out = 'example'
fn = r'ex1_black_BG.png'
chord_kwargs = {'black_BG': True}
plot_and_combine(dir_out, fn, idx_to_label, edges,
                 edge_weights=edge_weights, coords=power_coords,
                 network_order=network_order, network_colors=network_colors,
                 chord_kwargs=chord_kwargs, title='Example 1b (black)')
```
<p align="center">
  <img src="example\ex1_black_BG.png" width="800"/>
</p>

Here is another example. This one shows how setting `linewidth = 0` in the glass brain kwargs causes only the glass brain ROIs to be plotted. This may be useful in combination with setting a `node_size` as a list, which causes nodes on the glass brain to be plotted in sizes specified. 

```Python
fn = r'ex1_count.png'
chord_kwargs = {'plot_count': True}
n_nodes = len(set([i for i, j in edges] + [j for i, j in edges]))
glass_kwargs = {'linewidths': 0.,
                'node_size': list(range(1, n_nodes+1))}
plot_and_combine(dir_out, fn, idx_to_label, edges,
                 edge_weights=edge_weights, coords=power_coords,
                 network_order=network_order, network_colors=network_colors,
                 chord_kwargs=chord_kwargs, glass_kwargs=glass_kwargs,
                 title='Example 1c (count)')
```

<p align="center">
  <img src="example\ex1_count.png" width="800"/>
</p>


This next example shows further features. With `do_ROI_cicles=True`, you can plot little circles on the chord diagrams where the arcs start with. With `only1glass=True`, you can the sagittal glass brain only. These arguments are not specific to `plot_and_combine`. They can also be passed to `plot_chord` and `plot_glassbrain`, respectively.

```Python
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
                 chord_kwargs=chord_kwargs, glass_kwargs=glass_kwargs)
```

<p align="center">
  <img src="example\ex2.png"width="570"/>
</p>


### Convenience functions

For convenience, the function `convert.convert_matrix(matrix)` is provided, which takes a matrix as input and returns two lists corresponding to `edges` and `edge_weights`.

```Python
from nichord.convert import convert_matrix

matrix = [[0, 0.5, 0.2], [0.5, 0, -0.2], [0.2, -0.2, 0]]
edges, edge_weights = convert_matrix(matrix)
```


## Note
There is seemingly a bug in matplotlib.backend_agg.RenderAgg, which makes rotated text not look ideal when plotting character by character. I submitted a [bug report](https://github.com/matplotlib/matplotlib/issues/23021) to matplotlib, along with a potential
solution. It has not been accepted yet, so for now I am "monkey patching" the
malfunctioning code in `nichord.patch_RenderAgg.py`. The patch is automatically 
applied when calling `chord.plot_chord_diagram(...)` with the default argument 
`do_monkeypatch=True`. 

The glass brain diagrams rely on the plotting tools from [`nilearn`](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_connectome.html), whereas the chord diagrams is made from scratch by drawing shapes in [`matplotlib`](https://matplotlib.org/).

## Authors
`NiChord` was created by Paul C. Bogdan with help from [Jonathan Shobrook](https://github.com/shobrook) as part of our research in the [Dolcos Lab](https://dolcoslab.beckman.illinois.edu/) at the Beckman Institute for Advanced Science and Technology and the University of Illinois at  Urbana-Champaign. 

If you are using `NiChord` in your work, we ask that you please cite the paper below or the present reporsitory:

Bogdan, P.C., Iordan, A. D., Shobrook, J., & Dolcos, F. (2023). ConnSearch: A Framework for Functional Connectivity Analysis Designed for Interpretability and Effectiveness at Limited Sample Sizes. *NeuroImage*, 120274.