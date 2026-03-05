# Learning Basis Functions on Function Spaces

This is the repo related to my work on learning infinite basis functions on function space and parameterizing infinite-dimensional rotations. For more details on the theory of the method please look at the Overleaf document.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate infidictionary
```

## Notebooks

The codebase includes a suite of Jupyter notebooks located in the [`notebooks/`](./notebooks/) directory. These serve as the primary entry point for understanding the different components of the repository. Below is a guide to the essential notebooks:

1. **[Bases Intro](./notebooks/infinite_bases.ipynb):** Introduces the `infidictionary` API for generating analytical basis functions. The primary objective here is to use isometric deformations and rotations of these simple bases (e.g., Fourier) to represent complex bases within function spaces, in this notebook, we show how to sample and visualize the basic simple bases.

2. **[Domain Samplers](./notebooks/domain_samplers.ipynb):** Provides the API for generating points across various domains, supporting both stochastic (e.g., independent and identically distributed, stratified) and deterministic (regular grid) sampling strategies. Beyond that, this notebook also contains information about isometries that are primarily used to map between different geometries, such as applying the Shirley-Chiu or Polar transform to change between a disk domain and a square domain.

3. **[Neural Isometries](./notebooks/isometries/):** The notebooks in this directory form the core of our method. Here, we implement various deformations and isometries, including change-of-domain mappings (e.g., transforming a Fourier basis on a square to a sphere), Lagrangian continuous-time flows, Eulerian low-rank parameterizations, and mixed approaches. To navigate these concepts, we recommend following this natural progression:
   
   * **(i) [Eulerian Isometries](./notebooks/isometries/eulerian_transform.ipynb)**
     * Contains foundational experiments for working with the isometric transform API.
     * Focuses on fully Eulerian isometries, specifically those that discretize to Householder reflections using a skew-adjoint low-rank parameterization.
   
   * **(ii) [Normalizing Flows](./notebooks/isometries/normalizing_flow.ipynb)**
     * Defines isometries through a single diffeomorphism using the isometric change-of-variable trick, rather than an ODE.
     * In this notebook we explore complex mappings constructed via normalizing flows, including neural spline flows.
   
   * **(iii) [Lagrangian Isometries](./notebooks/isometries/ct_flows.ipynb)**
     * Constructs isometries using continuous-time flows based on the Lie derivative formulation.
   
   * **(iv) [Semi-Lagrangian Isometries](./notebooks/isometries/semi_lagrangian.ipynb)**
     * The culmination of the preceding methods, mixing the Eulerian and Lagrangian formulations.
     * Introduces a parameterized neural gating mechanism that dynamically determines the balance at any given point between the Lagrangian component, using the Lie derivative generator and the Eulerian component, using the low-rank skew-adjoint generator.

   * **(v) [Deforming Datasets](./notebooks/isometries/examples.ipynb)**
     * Visualizing the deformations applied to some of the datasets.

# Running Scripts

We use a combination of [Weights & Biases](https://docs.wandb.ai/models/quickstart) for logging our experiments and [Hydra](https://hydra.cc/) for tracking our configurations. Please read their corresponding tutorials for setting up.

## Functional PCA

The `fpca.py` script tries to fit a basis that captures the first and second order characteristics of a functional dataset (refer to the Overleaf for more information). Here are some example usages:

```bash
python fpca.py +experiment=sanity_check_eulerian
python fpca.py +experiment=random_bandpass_eulerian wandb=enabled wandb.run_name=random_bandpass_eulerian
python fpca.py +experiment=random_bandpass_eulerian_disk wandb=enabled wandb.run_name=random_bandpass_eulerian_disk
# Lagrangian isometries
python fpca.py +experiment=sanity_check_lagrangian_disk
python fpca.py +experiment=random_bandpass_lagrangian_disk wandb=enabled wandb.run_name=random_bandpass_lagrangian_disk
# SemiLagrangian isometry
python fpca.py +experiment=semilagrangian_disk
```

**Resuming an Experiment**: To resume an experiment try running the following:
```bash
python fpca.py +experiment=<experiment-config> resume_training.enabled=true resume_training.checkpoint_path=<path-to-checkpoint>
```

## Fourier-er

TODO