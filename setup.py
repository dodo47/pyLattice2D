import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyLattice2D",
    version="1.0",
    author="Dominik Dold, Derek Aranguren van Egmond",
    author_email="dodo.science@web.de",
    description="Library for generating and inverse-designing a variety of differentiable 2D lattice tilings.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.7',
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'seaborn',
          'jupyter',
          'networkx',
          'dgl',
          'torch',
          'tqdm',
          'sortedcontainers',
          'shapely',
          'scikit-learn',
    ]
)
