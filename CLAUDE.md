# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda env create -f environment.yml
conda activate infidictionary
```

## Running Experiments

All scripts use [Hydra](https://hydra.cc/) for config composition and [W&B](https://docs.wandb.ai/) for logging.

**Functional PCA** (`fpca.py`) — fits a basis capturing first/second-order statistics of a functional dataset:
```bash
python fpca.py +experiment=random_bandpass_eulerian
python fpca.py +experiment=random_bandpass_eulerian_disk wandb=enabled wandb.run_name=my_run
python fpca.py +experiment=sanity_check_lagrangian_disk
```

**Concept Basis** (`concept_basis.py`) — steers a basis toward a text concept via SDS or CLIP:
```bash
python concept_basis.py +experiment=sds_cat
python concept_basis.py +experiment=clip_cat
```

**Resuming a run:**
```bash
python fpca.py +experiment=<name> resume_training.enabled=true resume_training.checkpoint_path=<path>
```

Hydra overrides work inline: append `key=value` to any command. The `eval` resolver is registered so YAML can use `${eval:'expr'}`.

## Architecture

The codebase implements *infinite-dimensional isometric learning*: a `NeuralIsometry` is trained to rotate an initial analytic dictionary (e.g., Fourier) into a new basis that better captures a dataset, while preserving all inner products.

### Core abstractions (`infidictionary/`)

**`NeuralIsometry`** (`neural_isometries/base.py`) — abstract `nn.Module` with two methods:
- `pushforward(src_coords, src_logabsdet, src_field)` → `(tgt_coords, tgt_logabsdet, tgt_field)`
- `pullback(tgt_coords, tgt_logabsdet, tgt_field)` → `(src_coords, src_logabsdet, src_field)`

All tensors follow the convention: coords `(N, d)`, logabsdet `(N,)`, field `(B, N, C)`.

Concrete implementations:
- **`EulerianIsometry`** — coordinate map is the identity; rotates only function values via a sequence of Householder reflections parameterized by a `TimeEvolvingField`. Cheap to invert (just reverse the time span).
- **`LagrangianIsometry`** (`continuous_time.py`) — isometry through a diffeomorphism; both coordinates and values transform.
- **`SemiLagrangianIsometry`** — gated mix of Eulerian and Lagrangian components.
- **`NormalizingFlowIsometry`** (`normalizing_flows.py`) — isometry via a single diffeomorphism (neural spline flow).

**`InfiDictionary`** (`dictionaries/base.py`) — abstract collection of atoms (basis functions). Key methods:
- `get_atoms(coords, idx)` → `(A, N, C)`: evaluate atoms at quadrature points.
- `monte_carlo_captured_energy(coords, logabsdet, values, ...)`: stratified estimator splitting into an exact high-probability stratum and an MC tail.
- `get_truncated_indices(K)`: deterministic finite sub-dictionary for reconstruction/rendering.

Concrete: `FourierDictionary2D` (Fourier on square/disk).

**`NeuralField`** (`networks/`) — neural network mapping `(N, d)` coords to `(N, C)` values. Variants include `FourierFeatureField`, `NTKMLPField`, `TimeEvolvingField`, `FourierMixtureField`.

**`DomainSampler`** (`domain_samplers.py`) — samples quadrature points on a domain (square, disk), supporting iid, stratified, or grid strategies.

**`ConceptLoss`** (`concept.py`) — `SDSLoss` (DreamFusion-style Score Distillation Sampling via frozen Stable Diffusion) and `CLIPLoss` (negative cosine similarity via OpenCLIP). Both share a two-call interface: `encode_text(prompt, device)` once, then `loss(images, text_encoded)` every step.

### Training scripts

`fpca.py` trains a `NeuralIsometry` + `NeuralField` (mean function) jointly:
1. Mean function minimizes MSE to the dataset mean.
2. Isometry maximizes `monte_carlo_captured_energy` of the zero-centered data pulled back to the source domain.

`concept_basis.py` trains an isometry so that random sparse linear combinations of pushed-forward atoms render like a target text concept. Uses gradient accumulation, optional mean function, and `variation_strength` to scale the isometry contribution.

### Config layout (`conf/`)

Each experiment YAML composes from sub-configs in:
- `conf/neural_isometry/` — isometry type and architecture
- `conf/neural_fields/` — mean function / field network
- `conf/dictionaries/` — initial dictionary
- `conf/diffeomorphisms/` — diffeomorphism for Lagrangian/normalizing-flow isometries
- `conf/domain_samplers/` — domain and sampling strategy
- `conf/energy_estimation_kwargs/` — `monte_carlo` or `nufft`
- `conf/callbacks/` — visualizations, isometry checks, reconstructions
- `conf/fpca_experiment/`, `conf/concept_experiment/` — full experiment overrides

Checkpoints are saved under `outputs/checkpoints/<run_name>/` (Hydra output dir). W&B run ID is embedded in the run name as `wandb-<id>` to support resumption.
