import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 5, 0):
    print("Requires Python 3.5 to run.")
    sys.exit(1)

desc = '`NiChord` is a Python package for visualizing functional connectivity data. This package was inspired by [NeuroMArVL](https://immersive.erc.monash.edu/neuromarvl/?example=40496078-effa-4ac3-9d3e-cb7f946e7dd1_137.147.133.145), an online visualization tool.\n\n' + \
'The code can function with any configuration of edges and labels specified by the user.\n\n' + \
'The glass brain diagrams (left & middle) rely on the plotting tools from [nilearn](https://nilearn.github.io/modules/generated/nilearn.plotting.plot_connectome.html), whereas the chord diagram (right) is made from scratch by drawing shapes in [matplotlib](https://matplotlib.org/). Most of the code, here, is dedicated to the chord diagrams.\n\n' + \
'This package additionally provides code to help assign labels to nodes based on their anatomical location.\n\n' + \
'To find out more about the package and see an example, see its [GitHub repo](https://github.com/paulcbogdan/NiChord).'

setup(
    name="nichord",
    description="Creates chord diagrams for connectivity/graph data",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="v0.1.2",
    packages=["nichord"],
    python_requires=">=3.5",
    url="https://github.com/paulcbogdan/NiChord",
    author="paulcbogdan",
    author_email="paulcbogdan@gmail.com",
    install_requires=["nilearn", "numpy", "matplotlib", "scipy", "atlasreader",
                      "pillow"],
    keywords=["plotting", "fmri", "plotting", "chord"],
    license="MIT"
)