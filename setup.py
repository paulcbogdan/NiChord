import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.version_info[:3] < (3, 7, 0):
    print("Requires Python 3.7 to run.")
    sys.exit(1)

desc = '`NiChord` is a Python package for visualizing functional connectivity data. To find out more about the ' \
       'package and see a package, take a look at its [GitHub repo](https://github.com/paulcbogdan/NiChord)'

'To find out more about the package and see an example, see its .'

setup(
    name="nichord",
    description="Creates chord diagrams for connectivity/graph data",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="v0.2.2",
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
                      "pillow",
                      "atlasreader"],
    keywords=["plotting", "fmri", "plotting", "chord"],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    license="MIT"
)