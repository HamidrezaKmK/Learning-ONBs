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

### Regularizer-based change-of-basis (`reg_cob.py`)

The third training mode learns a basis by minimising a geometric energy rather than fitting data. The entry point is `reg_cob.py`; config root is `conf/reg_cob.yaml`.

```bash
python reg_cob.py +vibration_experiment=airplane_vibration
python reg_cob.py +vibration_experiment=trivial_vibration
python reg_cob.py +tv_experiment=1d_tv
```

**Inner loop** (each epoch):
1. `regularizer.update_coordinates(neural_isometry, pushforward_kwargs)` — samples fresh quadrature points and pre-computes their pullback. Subclasses also cache material values (diffusivity, mass, KNN graphs) here.
2. `grad_accumulation_steps` micro-steps, each with a fresh `shuffle_model_state` call (re-samples the ODE time-span for Eulerian isometries).
3. Atoms are split into a *high-probability exact stratum* (deterministic, weighted by `pmf`) and a *tail stratum* (MC, weighted by sample count). Both are back-propagated and the gradients are accumulated.
4. Optional gradient clipping (`max_grad_norm`), then `optim_isometry.step()`.

**`_eval_pushed_atoms`** (`regularizers/base.py`) — shared helper used by every `compute_energy` implementation:
- Pullback is skipped when pre-computed `src_coords`/`src_logabsdet` are passed in (saves a forward pass).
- For `EulerianIsometry` the pullback is the identity; `src_coords = tgt_coords`, `src_logabsdet = 0`.
- Gradients flow only through the pushforward, not the (detached) pullback.

### Materials (`material.py`)

`Material` is an abstract base class with three responsibilities:

| Method | Shape | Purpose |
|---|---|---|
| `__call__(coords)` | → `(diffusivity (N,), mass (N,))` | Evaluate D(x) and ρ(x) |
| `project_to_domain(coords)` | `(…, d)` → `(…, d)` | Fold coordinates back onto the canonical domain |
| `neighbourhood(coords, K, h)` | `(N,d)` → 3× `(N,K,d)` | K random symmetric FD pairs `(x+hδ, x−hδ, δ)` |
| `diffuse(coords, K, num_steps, dt, h_fd)` | `(N,d)` → `(N,K,d)` | Itô Euler–Maruyama trajectories of the SDE `dX=(∇D/ρ)dt + √(2D/ρ)dW` |

`neighbourhood` and `diffuse` live in the base class and call `self.project_to_domain` for wrapping — domain topology is encapsulated in each subclass, not hardcoded.

Concrete materials:
- **`FourierTorusMaterial`** — D(x) = base + amplitude·cos(2π Σ kᵢ(xᵢ−φᵢ) + θ). Setting `amplitude=0` gives a spatially uniform (constant) diffusivity. Torus wrapping via `% 1.0`.
- **`JosephFourierMaterial`** — diffusivity sampled from a greyscale portrait; supports Gaussian pre-smoothing and a binary mass mask via `threshold`.
- **`AirplaneMaterial`** — analytic silhouette (body + wing + tail) with `interior_diffusivity` (structure) and `exterior_diffusivity` (background). The fill order is: initialise all to `exterior_diffusivity`, then set the inside mask to `interior_diffusivity`.

Key naming convention: `interior_diffusivity` = inside the shape (was historically `max_diffusivity`), `exterior_diffusivity` = background (was `min_diffusivity`).

### Regularizers (`regularizers/`)

All regularizers inherit from `Regularizer` (or `PushforwardRegularizer`). They own a `DomainSampler` and expose two methods: `update_coordinates` (called once per epoch, no grad) and `compute_energy` (called each micro-step, with grad).

| Class | Energy | Use case |
|---|---|---|
| `HeatQuadraticFormRegularizer` | −⟨Qφ, P̃_t Qφ⟩_ρ | Vibrational modes; maximises the heat-kernel quadratic form. P̃_t is approximated via `material.diffuse(K=n_diffusion_samples, num_steps=1, dt=smoothing_t)`. Optional `masking_weight` penalises atom values outside the material support. |
| `DirichletEnergyRegularizer` | ∫ D‖∇(Qφ)‖² dx | Weighted Dirichlet energy via stochastic FD. Neighbours come from `material.neighbourhood(K=1)`. Multiply-by-d corrects for the E[(∇f·δ)²]=(1/d)‖∇f‖² identity. |
| `TVMaterialRegularizer` | Σ_{edges} w_e ‖Qφ(i)−Qφ(j)‖ | Sparse TV / 1-Laplacian basis. Uses a **faiss** KNN graph (built in `update_coordinates`) with Gaussian edge weights; only inside-material edges contribute. Inherits `PushforwardRegularizer` so the pullback/pushforward is handled automatically. |
| `FouriererRegularizer` | vibration + masking + diversity | Two-channel specialisation: channel 0 atoms are pushed toward the interior vibrational modes, channel 1 toward the exterior. Includes a channel-equalisation term to prevent collapse. |
| `NTKRegularizer` | −Σ ‖∇_θ⟨Q*f_θ, φ_a⟩‖² | NTK quadratic form via implicit Jacobian; `create_graph=True` lets gradients reach the isometry. The NTK model is kept frozen. |
| `EuclideanGroup` | ‖Qφ(x) − φ(Rx−t)‖² | Symmetry regularizer: pushes atoms to be invariant to a fixed translation + rotation + mirror of the domain. Wraps transformed coordinates with `% 1.0`. Note: currently only tested with Eulerian isometries. |
