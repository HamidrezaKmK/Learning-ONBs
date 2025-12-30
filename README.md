# Learning Basis Functions on Function Spaces

This project is aimed at learning basis functions on function spaces using diffeomorphic transformations. For more details on the theory of the method please look at the Overleaf document.

The code is implemented in Python and leverages PyTorch for automatic differentiation and optimization.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate basis_learning
```

## Notebooks

The codebase comes with a set of Jupyter notebooks, all stored in the [`notebooks/`](./notebooks/) directory that act as a starting point for understanding different parts of the project. Here is a brief explanation of the notebooks: 

1. Examples Bases: This notebook constructs different basis functions on different domains and is a good visualization of the types of domains and function spaces we are dealing with.
2. Reconstruction: This notebook contains an example low-dimensional function space that is constructed by taking linear combinations of trigonometric functions with random phase and frequencies.

