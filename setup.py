import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 5, 0):
    print("Requires Python 3.5 to run.")
    sys.exit(1)

with open("setup_desc.txt", encoding="utf-8") as file:
    desc = file.read()

setup(
    name="nichord",
    description="Creates chord diagrams for connectivity/graph data",
    long_description=desc,
    long_description_content_type="text/markdown",
    version="v0.0.2",
    packages=["nichord"],
    python_requires=">=3.5",
    url="https://github.com/paulcbogdan/NiChord",
    author="paulcbogdan",
    author_email="paulcbogdan@gmail.com",
    install_requires=["nilearn", "numpy", "matplotlib", "scipy", "atlasreader"],
    keywords=["plotting", "fmri", "plotting", "chord"],
    license="MIT"
)