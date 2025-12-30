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

1. [Examples Bases](./notebooks/examples_bases.ipynb): This notebook constructs different basis functions on different domains and is a good visualization of the types of domains and function spaces we are dealing with.
2. [Reconstruction](./notebooks/reconstruction.ipynb): This notebook contains an example low-dimensional function space that is constructed by taking linear combinations of trigonometric functions with random phase and frequencies.
3. [Normalizing Flow](./notebooks/normalizing_flow.ipynb): This notebook is our first exploration of parameterizing a change of basis via diffeomorphisms. We parameterize diffeomorphisms using normalizing flows (e.g., neural spline flows) and study how they warp a previously defined basis. The main goal is to familiarize the reader with the transformation operator in practice and to empirically validate the theoretical results from the write-up.
4. Continuous Time Flows: Once the first class of diffeomirphisms is covered, we go over the continuous time flows that are defined through ordinary differential equations.
