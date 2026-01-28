# Learning Basis Functions on Function Spaces

(TODO: revamp the entire thing)

This project is aimed at learning basis functions on function spaces using diffeomorphic transformations. For more details on the theory of the method please look at the Overleaf document.

The code is implemented in Python and leverages PyTorch for automatic differentiation and optimization.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate infidictionary
```

## Notebooks

The codebase comes with a set of Jupyter notebooks, all stored in the [`notebooks/`](./notebooks/) directory that act as a starting point for understanding different parts of the project. Here is a brief explanation of the notebooks: 

1. [Examples Dictionary](./notebooks/examples_dictionaries.ipynb): This notebook constructs different basis functions on different domains and is a good visualization of the types of domains and function spaces we are dealing with.
2. [Reconstruction](./notebooks/reconstruction.ipynb): This notebook contains an example low-dimensional function space that is constructed by taking linear combinations of trigonometric functions with random phase and frequencies.
3. [Normalizing Flow](./notebooks/normalizing_flow.ipynb): This notebook is our first exploration of parameterizing a change of basis via diffeomorphisms. We parameterize diffeomorphisms using normalizing flows (e.g., neural spline flows) and study how they warp a previously defined basis. The main goal is to familiarize the reader with the transformation operator in practice and to empirically validate the theoretical results from the write-up.
4. [Continuous Time Flows](./notebooks/ct_flows.ipynb): Once the first class of diffeomirphisms is covered, we go over the continuous time flows that are defined through ordinary differential equations.

# Running Scripts

We use a combination of [Weights & Biases](https://docs.wandb.ai/models/quickstart) for logging our experiments and [Hydra](https://hydra.cc/) for tracking our configurations. Please read their corresponding tutorials for setting up.

## Reconstruction

The `reconstruction.py` script is the first script that actually optimizes a diffeomorphism for the tasks of reconstruction. Here are a few experiment templates:

```bash
python reconstruction.py +experiment=kumaraswamy_sanity_check # (1)
python reconstruction.py +experiment=spline_sanity_check # (2)
python reconstruction.py +experiment=ct_radial_sanity_check # (3)
python reconstruction.py +experiment=spline_radial_random_bandpass #(4)
python reconstruction.py +experiment=ct_radial_random_bandpass # (5)
```

1. A very simple sanity check where a Kumaraswamy diffeomorphism is trained to match a dataset generated from a random combination of the initial basis. The optimal solution to this problem is the identity diffeomorphism.
2. Another sanity check similar to (1) where a spline flow is trained instead of a Kumaraswamy map. This is to sanity check the spline flow.
3. Similar to (1) and (2) but with a continuous-time normalizing flow (CNF) instead. The optimal map would have zero velocity fields, i.e., the identity flow map. Another change compared to (1) and (2) is that here a basis on the unit disk is used.
4. In this experiment a spline flow is trained to match a synthetic dataset of trigonomic functions (the one described in the reconstruction notebook) on the radial basis.
5. Similar to (4) but rather than using a spline flow we use a CNF.

**Note:** For the reconstruction code to properly work, the batch size typically needs to be set to a large value.
