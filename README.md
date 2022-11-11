# NiChord

`NiChord` is a Python package for visualizing functional connectivity data. 
This package was inspired by [NeuroMArVL](https://immersive.erc.monash.edu/neuromarvl/?example=40496078-effa-4ac3-9d3e-cb7f946e7dd1_137.147.133.145),
an online visualization tool. With just a few lines of Python code, `NiChord`
 can create figures like these:
  
 <p align="center">
  <img src="example\ex_1_and_2.png" />
</p>

The code can function with any configuration of edges and labels specified by
the user. 

The glass brain diagrams (left & middle) rely on the plotting tools from 
[nilearn](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_connectome.html), 
whereas the chord diagram (right) is made from scratch by drawing shapes in 
[matplotlib](https://matplotlib.org/). Most of the code, here, is dedicated to 
the chord diagrams. 
  
 This package additionally provides code to help assign labels to nodes based
 on their anatomical location. 

## Installation
`NiChord` (requires Python 3.5+) can be installed via pip:

```
$ pip install nichord
```

## Usage example

The examples above were constructed by saving separate images for the chord and
glass brain diagrams and then combining the images.

### Input variables
Edges are specified as a list of tuples, (i, j), where i and j are indices
representing the two nodes making up the edge.
```
edges = [(0, 1), (0, 2), (1, 5), (3, 5), (4, 6), (2, 7), (6, 7)]
```

Each node index should also correspond to a coordinate in MNI space:
```
coords = [[-24, -99, -12], [51, -3, -15], [-15, -70, 30], [21, 39, 39],
          [21, -66, 48], [54, 33, 12], [-33, 3, 3], [57, -45, 12]]
```

These coordinates can be used to construct a dictionary, mapping each node index
to a network label (by default, network labels are based on the 
[Yeo et al. (2011) atlas](https://journals.physiology.org/doi/full/10.1152/jn.00338.2011)):

```
from nichord.coord_labeler import get_idx_to_label
idx_to_label = get_idx_to_label(coords, atlas='yeo')
```

Or a dictionary can be defined manually:

```
idx_to_label = {0: 'Visual', 1: 'DMN', 2: 'Visual', 3: 'DMN', 
                4: 'DAN', 5: 'FPCN', 6: 'VAN', 7: 'VAN'}
```

Each edge may be associated with a weight. Weights are defined as a list of 
length equal to the number of edges (if `edge_weights = None`, then grey edges are.
plotted).

```
edge_weights = [-0.3, -0.5, 0.7, 0.5, -0.2, 0.3, 0.8]
```

### Plotting

These variables and a filepath can then be 
passed to create the chord diagram:
```
from nichord.chord import plot_chord

fp_chord = 'ex0_good.png' # if None, chord diagram can be opened in a matplotlib
                          # window with matplotlib.pyplot.show()
plot_chord(idx_to_label, edges, edge_weights=edge_weights, 
    fp_chord=fp_chord,
    linewidths=15, alphas = 0.9, do_ROI_circles=True, 
    do_ROI_circles_specific=True, ROI_circle_radius=0.02)
```

<p align="center">
  <img src="example\ex0_chord.png" width="600" />
  <br>
  If no filepath is passed, the diagram will be opened in a matplotlib window.
</p>


Plotting the glass brain involves the same variables (note that the colors of 
the glass brain nodes should correspond to the same colors as the chord network 
labels)
```
from nichord.glassbrain import plot_glassbrain

fp_glass = 'ex0_glassbrain.png'
plot_glassbrain(idx_to_label, edges, edge_weights, fp_glass,
                coords, linewidths=15, node_size=17)
```

<p align="center">
  <img src="example\ex0_glass.png" />
</p>

Finally, to combine the figures above:
```
from nichord.combine import combine_imgs

fp_combined = 'ex0_combined.png'
combine_imgs(fp_glass, fp_chord, fp_combined)
```

Notably, these functions have many other optional variables (e.g., passing
specific colors for each network label using a dictionary). Further information
on these can be seen by examining `example\example.py`, which contains code used
to generate the two examples from the first section of this README. 
You can also learn about these by reading the hinting/documentation within each 
function.


<p align="center">
  <img src="example\ex0_combined.png" />
</p>


### Convenience functions

For convenience, the function `convert.convert_matrix(matrix)` is provided, which
takes a matrix as input and returns two lists corresponding to edges and 
edge_weights.

```
from nichord.convert import convert_matrix

matrix = [[0, 0.5, 0.2], [0.5, 0, -0.2], [0.2, -0.2, 0]]
edges, edge_weights = convert_matrix(matrix)
```

### Note
There is seemingly a bug in matplotlib.backend_agg.RenderAgg, which makes 
rotated text not look ideal when plotting character by character, as is being
done here. I submitted a [bug report](https://github.com/matplotlib/matplotlib/issues/23021)
to matplotlib, along with a potential
solution. It has not been accepted yet, so for now I am "monkey patching" the
malfunctioning code in `nichord.patch_RenderAgg.py`. The patch is automatically 
applied when calling `chord.plot_chord_diagram(...)` unless 
`do_monkeypatch=False`. 

## Update (November 10, 2022)
Additions:
* updated to allow titles for combine_image (new title= argument; '' by default)
* updated to allow black_BG for plot_chord (new black_BG= argument; false by default, which causes a white BG). See the example below. Note: black_BG=true with the new 'turbo' default looks much better for extremely dense chord diagrams (1000+ edges)
<p align="center">
  <img src="example\ex1_chord_black_BG.png"  width="600" />
</p>
<p align="center">
  <img src="example\outside_chord_example.png"  width="600" />
</p>
* updated to allow plot_chord to average all edges between a pair of networks (new plot_count= argument; default is false). Helpful when there are a lot of edges otherwise. See the example below, which is a plot_count=True version of the chord diagram immediately above. For plot_count=True, the arc color is the average of all edge weights for edges between a pair of networks. Arc thickness corresponds to the number of edges between the pair of networks
<p align="center">
  <img src="example\outside_chord_example_count.png"  width="600" />
</p>

Changed:
* Default plotting color from matplotlib's 'Spectral_r' to its 'turbo'. The middle end is much more visible now and the colors overall pop more
* Plot_chord to now plots the arcs in order of the absolute value of edge_weights or for plot_count in order of the count (thicker = on top)
* The hinting for plot_glassbrain to specify that node_size can be a list or ndarray. Passing a list or ndarray of length = #nodes or #nonzero_nodes will cause the nodes to be plotted the different sizes specified

## Authors
`NiChord` was created by Paul C. Bogdan with help from [Jonathan
 Shobrook](https://github.com/shobrook) as part of our research in the 
 [Dolcos Lab](https://dolcoslab.beckman.illinois.edu/) at the Beckman Institute
 for Advanced Science and Technology and the University of Illinois at 
 Urbana-Champaign. 