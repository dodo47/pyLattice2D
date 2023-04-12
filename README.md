# pyLattice2D

`pyLattice2D` is a Python package based on PyTorch and DGL (Deep Graph Library) for generating 2D lattices, performing finite element analysis (more specifically, direct stiffness with generalized Euler Bernoulli beams) and inverse designing 2D lattice materials. It features a differentiable graph-based model of lattices that allows the usage of automatic differentiation to change a lattice's geometry and material properties (e.g., remove or add beams).

This package was created as part of the publication "Differentiable graph-structured models for inverse design of lattice materials". An introduction to its structure is provided in the publication's supplementary information.

## Code

### Installation

To use our code, install the provided package pyLattice2D via pip
```
pip install -e .
```
from the folder containing the `setup.py`. 
The source code is located in `pyLattice2D/`. 

### Examples

Example notebooks replicating some of the experiments shown in the publication can be found in `Examples/`. 

### Features

Included methods for generating (and perturbing, i.e., by moving/removing nodes, adding/removing beams) base tilings:
- square
- (equilat.) triangle
- honeycomb
- reentrant honeycomb
- Kagome
- Voronoi (with different degrees of disorder)

The direct stiffness method allows:
- calculation of deformations to general loads (iteratively)
- measurement of effective elastic modulus
- measurement of Poisson's ratio

The code also includes graph neural network models as well as tabular models (trainable on tabular features characterizing a lattice material) and a convolutional neural network (trainable on images of lattices) to predict material properties.

To enable using gradient descent for adding/removing beams during inverse design, we add masking values for each beam. A beam is realized in the material if the masking value is larger than 0 (i.e., the masks are thresholded at 0). To improve training speed, a method called `Surrogate Gradient` is used when calculating gradients through this threshold function.

The implemented direct stiffness method and the graph neural networks are based on message passing, and hence allow inverse design via automatic differentiation on features of the input lattice (beam existence, beam cross-area, beam stiffness, node positions, etc.). In principle, this enables the inverse design of completely heterogeneous lattice structures to discover materials that possess certain desired properties.

## Abstract of the accompanying publication

Materials possessing flexible physico-chemical properties that adapt on-demand to the hostile environmental conditions of deep space will become essential in defining the future of space exploration. A promising venue for inspiration towards the design of environment-specific materials is in the intricate micro-architectures and lattice geometry found throughout nature. However, the immense design space covered by such irregular topologies is challenging to probe analytically. For this reason, most synthetic lattice materials have to date been based on periodic architectures instead. Here, we propose a computational approach using a graph representation for both regular and irregular lattice materials. Our method uses differentiable message passing algorithms to calculate mechanical properties, and therefore allows using automatic differentiation to adjust both the geometric structure and attributes of individual lattice elements to design materials with desired properties. The introduced methodology is applicable to any system representable as a heterogeneous graph, including other types of materials.

## Citation

If you use the provided code or find it helpful for your own work, please cite

```
@article{dold2023differentiable,
  title={Differentiable graph-structured models for inverse design of lattice materials},
  author={Dold, Dominik and Aranguren Van Egmond, Derek},
  journal={},
  volume={},
  number={},
  pages={},
  year={2023},
  publisher={}
}
```