import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info[:3] < (3, 7, 0):
    print("Requires Python 3.7 to run.")
    sys.exit(1)

desc = '`NiChord` is a Python package for visualizing functional connectivity data. To find out more about the ' \
       'package and see a package, take a look at its [GitHub repo](https://github.com/paulcbogdan/NiChord). ' \
       'The package is also available via [conda](https://anaconda.org/conda-forge/nichord).'

setup(
    name="nichord",
    description="Creates chord diagrams for connectivity/graph data",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="v0.3.3",
    packages=["nichord"],
    python_requires=">=3.7",
    url="https://github.com/paulcbogdan/NiChord",
    author="paulcbogdan",
    author_email="paulcbogdan@gmail.com",
    install_requires=["nilearn",
                      "pandas",
                      "matplotlib",
                      "numpy",
                      "scipy",
                      "pillow"],
    keywords=["plotting", "fmri", "plotting", "chord"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    license="MIT",
    license_file="LICENSE.txt"
)