"""vibrational_modes.py

Train a NeuralIsometry Q to find the vibrational modes of a material.

Given a material with spatially varying diffusivity D(x), the vibrational modes
are the eigenfunctions of the weighted Laplacian  -∇·(D(x) ∇).  The k-th mode
minimises the weighted Dirichlet energy

    E_k[f] = ∫ D(x) ‖∇f(x)‖² dx

subject to f being L²-orthogonal to the first k-1 modes.

We find all modes simultaneously by training Q to minimise the expected Dirichlet
energy over atoms drawn from the prior:

    L(Q) = E_{i ~ pmf} [ ∫ D(x) ‖∇(Q φ_i)(x)‖² dx ]

where φ_i are the Fourier basis atoms on the source domain.  An isometry Q that
perfectly diagonalises the weighted Laplacian minimises this objective and maps
Fourier atoms to vibrational modes.

The Dirichlet energy equals the inner product

    ∫ D(x) ‖∇(Qφ_i)(x)‖² dx = ⟨∇(Qφ_i), D · ∇(Qφ_i)⟩_{L²}

The gradient ∇(Qφ_i) is a (C × d) tensor field; treating C*d as channels lets
us use parallel_inner_product directly.

Spatial gradient computation — unified for Eulerian and Lagrangian isometries:
  1. Sample target-domain points tgt_coords with requires_grad=True (leaf).
  2. Pull back through Q:  src_coords, logabsdet_src = pullback(tgt_coords).
       EulerianIsometry:   src_coords = tgt_coords,         logabsdet_src = 0.
       LagrangianIsometry: src_coords = T⁻¹(tgt_coords),   logabsdet_src = log|J_{T⁻¹}|.
  3. Evaluate atoms:  φ_src = φ(src_coords)  — depends on tgt_coords via src_coords.
  4. Push forward:    Qφ = pushforward(src_coords, logabsdet_src, φ_src)
                     — depends on tgt_coords via src_coords (and velocity fields).
  5. The Jacobian of  (Σ_j Qφ[a,j,c])  w.r.t. tgt_coords has shape (A*C, N, d);
     vectorize=True batches all A*C backward passes via vmap, mirroring the
     vectorised NTK computation in ntk_diagonalize.py.

Atom indices are drawn with the same stratified scheme as ntk_diagonalize.py:
  - Exact stratum:  all indices with pmf ≥ tail_probability, weighted by pmf.
  - MC tail stratum: num_tail_samples random indices (excluding exact stratum),
                     weighted by counts / num_tail_samples.
"""

import datetime
import os

import matplotlib
matplotlib.use("Agg")
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from infidictionary.checkpointing import Checkpointer
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.material import Material
from infidictionary.neural_isometries import NeuralIsometry, EulerianIsometry
from infidictionary.utils import parallel_inner_product
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

OmegaConf.register_new_resolver("eval", eval)


def dirichlet_energy_for_atoms_finite_difference(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    tgt_coords: torch.Tensor,       # (N, d) detached
    diffusivity: torch.Tensor,      # (N,) detached
    unique_indices: torch.Tensor,   # (A, domain_dim+1)
    pushforward_kwargs: dict,
    finite_diff_h: float = 1e-3,    # step size for finite difference
    n_fd_dirs: int = 1,             # number of random directions to average over
    **kwargs,
) -> torch.Tensor:                  # (A,)
    """Compute the weighted Dirichlet energy for each atom via stochastic finite diff.

    For atom a, estimates:
        DE_a ≈ (d / n_fd_dirs) Σ_{k=1}^{n_fd_dirs}
                    ⟨(Qφ_a(x+hδ_k) - Qφ_a(x-hδ_k))/(2h),
                     D · (Qφ_a(x+hδ_k) - Qφ_a(x-hδ_k))/(2h)⟩_{L²}
    where each δ_k is sampled independently and uniformly from S^{d-1}.

    Each direction uses a centered difference (two evaluations, no shared base
    point), so gradient contributions from different directions are fully
    independent and variance scales cleanly as 1/n_fd_dirs.

    Args:
        neural_isometry:    The isometry Q.
        initial_dictionary: Fourier atom dictionary.
        tgt_coords:         Target-domain quadrature points (detached).
        diffusivity:        D(x_j) at each point (detached).
        unique_indices:     Atom multi-indices, shape (A, domain_dim+1).
        pushforward_kwargs: Passed to pullback and pushforward.
        finite_diff_h:      Step size h for the finite difference derivative.
        n_fd_dirs:          Number of random directions to average over.

    Returns:
        de: Dirichlet energies of shape (A,).
    """
    # remove the gradient from tgt_coords
    tgt_coords = tgt_coords.detach()

    A = unique_indices.shape[0]
    N, d = tgt_coords.shape
    device = tgt_coords.device
    dtype = tgt_coords.dtype

    def _eval_qphi(coords_: torch.Tensor) -> torch.Tensor:
        """Helper to evaluate Qφ at arbitrary coordinates."""
        N_ = coords_.shape[0]
        zeros_ = torch.zeros(N_, device=device, dtype=dtype)
        dummy_field = torch.zeros(1, N_, 1, device=device, dtype=dtype)

        if isinstance(neural_isometry, EulerianIsometry):
            # Eulerian case: pullback is identity, so src = tgt and logdet = 0
            src_ = coords_
            logdet_ = zeros_
        else:
            src_, logdet_, _ = neural_isometry.pullback(
                tgt_coords=coords_,
                tgt_logabsdet=zeros_,
                tgt_field=dummy_field,
                **pushforward_kwargs,
            )

        phi_ = initial_dictionary.get_atoms(src_, unique_indices)  # (A, N_, C)
        _, _, qphi_ = neural_isometry.pushforward(
            src_coords=src_,
            src_logabsdet=logdet_,
            src_field=phi_,
            **pushforward_kwargs,
        )  # (A, N_, C)
        
        return qphi_

    log_D = diffusivity.log()  # (N,)
    de_per_atom = torch.zeros(A, device=device, dtype=dtype)

    for _ in range(n_fd_dirs):
        # Sample a direction uniformly from S^{d-1}
        delta = torch.randn_like(tgt_coords)
        delta = delta / torch.norm(delta, dim=-1, keepdim=True)

        # Wrap both ±h shifts periodically into [0, 1)^d
        coords_plus  = tgt_coords + finite_diff_h * delta
        coords_plus  = coords_plus  - torch.floor(coords_plus)
        coords_minus = tgt_coords - finite_diff_h * delta
        coords_minus = coords_minus - torch.floor(coords_minus)

        qphi_plus  = _eval_qphi(coords_plus)
        qphi_minus = _eval_qphi(coords_minus)

        # Centered finite-difference directional derivative: (A, N, C)
        diff = (qphi_plus - qphi_minus) / (2.0 * finite_diff_h)

        # Accumulate: E[(∇f · δ)²] = (1/d)‖∇f‖², so multiply by d.
        de_per_atom = de_per_atom + parallel_inner_product(diff, diff, logabsdet=log_D)

    return de_per_atom * (d / n_fd_dirs)
 
def dirichlet_energy_for_atoms_with_autograd(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    tgt_coords: torch.Tensor,       # (N, d)  leaf with requires_grad=True
    diffusivity: torch.Tensor,      # (N,)   detached, no grad
    unique_indices: torch.Tensor,   # (A, domain_dim+1)
    pushforward_kwargs: dict,
    **kwargs,
) -> torch.Tensor:                  # (A,)
    """Compute the weighted Dirichlet energy for each atom — vectorised.

    For atom a:
        DE_a = ∫ D(x) ‖∇_x (Qφ_a)(x)‖² dx = ⟨∇(Qφ_a), D · ∇(Qφ_a)⟩_{L²}

    The full (A*C) × N × d Jacobian is computed in one call via
    torch.autograd.functional.jacobian with vectorize=True (vmap over the A*C
    backward passes), so there is no Python loop over atoms.

    Args:
        neural_isometry:    The isometry Q.
        initial_dictionary: Fourier atom dictionary with .num_channels attribute.
        tgt_coords:         Target-domain quadrature points, leaf requires_grad.
        diffusivity:        D(x_j) at each point (detached).
        unique_indices:     Atom multi-indices, shape (A, domain_dim+1).
        pushforward_kwargs: Passed to pullback and pushforward.

    Returns:
        de: Dirichlet energies of shape (A,).
    """
    A = unique_indices.shape[0]
    N, d = tgt_coords.shape
    C = initial_dictionary.num_channels
    device = tgt_coords.device
    dtype = tgt_coords.dtype

    def _qphi_channel_sums(tgt_coords_: torch.Tensor) -> torch.Tensor:
        """Return Σ_j Qphi[a,j,c] for each (atom, channel) pair.

        Shape: (A*C,).  The Jacobian of this function w.r.t. tgt_coords_ has
        shape (A*C, N, d), from which we extract the per-atom Dirichlet energy.
        """
        N_ = tgt_coords_.shape[0]
        zeros_ = torch.zeros(N_, device=device, dtype=dtype)
        dummy_field = torch.zeros(1, N_, 1, device=device, dtype=dtype)

        # Step 1 — inverse map: tgt_coords → src_coords
        #   Eulerian:   src = tgt,              logdet = 0
        #   Lagrangian: src = T⁻¹(tgt),        logdet = log|J_{T⁻¹}|
        src_, logdet_, _ = neural_isometry.pullback(
            tgt_coords=tgt_coords_,
            tgt_logabsdet=zeros_,
            tgt_field=dummy_field,
            **pushforward_kwargs,
        )

        # Step 2 — evaluate atoms at source coords (depends on tgt via src)
        phi_ = initial_dictionary.get_atoms(src_, unique_indices)  # (A, N_, C)

        # Step 3 — push forward to get Qphi (depends on tgt via src)
        _, _, qphi_ = neural_isometry.pushforward(
            src_coords=src_,
            src_logabsdet=logdet_,
            src_field=phi_,
            **pushforward_kwargs,
        )  # (A, N_, C)

        # Sum over spatial points — Jacobian will recover per-point gradients
        return qphi_.reshape(A * C, N_).sum(dim=-1)  # (A*C,)

    # Vectorised Jacobian: (A*C, N, d)
    # J[a*C+c, j, k] = ∂Qphi[a,j,c] / ∂tgt_coords[j,k]
    J = torch.autograd.functional.jacobian(
        _qphi_channel_sums, tgt_coords, create_graph=True, #vectorize=True,
    )  # (A*C, N, d)

    # Reshape to (A, N, C*d) treating (channel, spatial dim) as a flat channel axis
    # J_flat[a, j, c*d + k] = ∂Qphi[a,j,c] / ∂tgt_coords[j,k]
    J_flat = (
        J.reshape(A, C, N, d)   # (A, C, N, d)
         .permute(0, 2, 1, 3)   # (A, N, C, d)
         .reshape(A, N, C * d)  # (A, N, C*d)
    )

    # Dirichlet energy per atom: (1/N) Σ_j D(x_j) Σ_{c,k} J_flat[a,j,c*d+k]²
    #   = ⟨∇Qφ_a, D · ∇Qφ_a⟩_{L²}  with d*C treated as channels
    log_D = diffusivity.log()  # (N,)
    return parallel_inner_product(J_flat, J_flat, logabsdet=log_D)  # (A,)


def dirichlet_energy_for_atoms(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    tgt_coords: torch.Tensor,       # (N, d) detached
    diffusivity: torch.Tensor,      # (N,) detached
    unique_indices: torch.Tensor,   # (A, domain_dim+1)
    pushforward_kwargs: dict,
    method: str = "autograd",  # "autograd" or "finite_difference"
    **method_kwargs,
):
    if method == "autograd":
        return dirichlet_energy_for_atoms_with_autograd(
            neural_isometry, initial_dictionary,
            tgt_coords, diffusivity,
            unique_indices, pushforward_kwargs,
        )
    elif method == "finite_difference":
        return dirichlet_energy_for_atoms_finite_difference(
            neural_isometry, initial_dictionary,
            tgt_coords, diffusivity,
            unique_indices, pushforward_kwargs,
            **method_kwargs,
        )
    else:
        raise ValueError(f"Unknown method {method} for Dirichlet energy computation")
   

# ── Main training loop ────────────────────────────────────────────────────────


def vibrational_modes(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    material: Material,
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    optim_isometry: torch.optim.Optimizer,
    scheduler_isometry: torch.optim.lr_scheduler._LRScheduler | None,
    callbacks: list,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    tail_probability: float,
    num_tail_samples: int,
    model_state_kwargs: dict,
    pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
    method: str,
    dirichlet_energy_kwargs: dict,
    max_grad_norm: float | None = None,
):
    """Train Q to minimise the expected weighted Dirichlet energy.

    Per step:
      1. Sample target-domain coords from material.domain_sampler.
      2. Evaluate D(x) (detached — no gradient through the material).
      3. Mark coords as requires_grad=True for spatial differentiation.
      4. Draw atom indices using stratified sampling (exact + MC tail).
      5. Compute DE_a = ⟨∇Qφ_a, D·∇Qφ_a⟩ for each unique atom (vectorised).
      6. Accumulate the pmf-weighted expected energy and back-propagate.
    """
    neural_isometry = neural_isometry.to(device)
    optim_isometry.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        de_history_temp = []

        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # ── Sample target-domain quadrature points ────────────────────────
            coords_raw = material.domain_sampler.sample(domain_sample_size).to(device)

            # Evaluate diffusivity — detach so D(x) carries no gradient
            with torch.no_grad():
                diffusivity = material(coords_raw).detach()  # (N,)

            # Enable spatial gradients for Dirichlet energy computation
            tgt_coords = coords_raw.detach().requires_grad_(True)

            # ── Exact high-probability stratum ────────────────────────────────
            idx_exact = initial_dictionary.get_high_probability_indices(
                tail_probability
            ).to(device)
            de_exact_per_atom = dirichlet_energy_for_atoms(
                neural_isometry, initial_dictionary,
                tgt_coords, diffusivity,
                idx_exact, pushforward_kwargs,
                method=method, **dirichlet_energy_kwargs,
            )  # (A_exact,)
            pmfs_exact = initial_dictionary.get_index_pmfs(idx_exact).to(device)
            de_exact = (de_exact_per_atom * pmfs_exact).sum()

            # ── MC tail stratum ───────────────────────────────────────────────
            idx_all = initial_dictionary.sample_indices(num_tail_samples).to(device)
            in_exact = (
                idx_all[:, None, :] == idx_exact[None, :, :]
            ).all(dim=-1).any(dim=-1)
            idx_tail = idx_all[~in_exact]

            if idx_tail.shape[0] > 0:
                idx_tail_u, idx_tail_c = torch.unique(
                    idx_tail, return_counts=True, dim=0
                )
                de_tail_per_atom = dirichlet_energy_for_atoms(
                    neural_isometry, initial_dictionary,
                    tgt_coords, diffusivity,
                    idx_tail_u, pushforward_kwargs,
                    method=method, **dirichlet_energy_kwargs,
                )  # (A_tail,)
                de_tail = (
                    de_tail_per_atom * idx_tail_c.float()
                ).sum() / num_tail_samples
            else:
                de_tail = torch.zeros(1, device=device).squeeze()

            total_de = de_exact + de_tail
            (total_de / grad_accumulation_steps).backward(retain_graph=False)
            de_history_temp.append(total_de.item())

        de_item = sum(de_history_temp) / len(de_history_temp)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            wandb.log({"train/dirichlet_energy": de_item}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({"dirichlet_energy": de_item})
        optim_isometry.step()
        optim_isometry.zero_grad()
        step_scheduler(scheduler_isometry, de_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=de_item)

        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=None,
                wandb_enabled=wandb_enabled,
                device=device,
            )


# ── Hydra entry point ─────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="conf", config_name="vibrational_modes")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    material: Material = instantiate(conf.material)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, wandb_run_id = None, None
    if conf.resume_training.enabled and conf.resume_training.checkpoint_path is not None:
        checkpoint = torch.load(
            conf.resume_training.checkpoint_path, weights_only=False, map_location=device
        )
        run_name = checkpoint.get("run_name", "")
        if run_name and run_name.startswith("wandb-"):
            wandb_run_id = run_name[len("wandb-"):]
    else:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if conf.wandb.enabled:
        wandb_run_name = str(conf.wandb.run_name) if conf.wandb.run_name is not None else None
        tags = [f"{key}:{value}" for key, value in conf.wandb.tags.items()] if "tags" in conf.wandb else []
        wandb.init(
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf, resolve=True),
            tags=tags,
            settings=wandb.Settings(start_method="thread"),
            name=wandb_run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
        )
        run_name = f"wandb-{wandb.run.id}"
    elif wandb_run_id is not None:
        raise ValueError("You are resuming a wandb run without specifying wandb=enabled!")

    callbacks = (
        []
        if "callbacks" not in conf
        else [instantiate(cb) for cb in conf.callbacks.values()]
    )

    isometry_optimizer_callable = instantiate(conf.isometry_optimizer_callable)
    isometry_scheduler_callable = (
        instantiate(conf.isometry_scheduler_callable)
        if conf.get("isometry_scheduler_callable", None)
        else None
    )

    optim_isometry = isometry_optimizer_callable(neural_isometry.parameters())
    scheduler_isometry = (
        isometry_scheduler_callable(optim_isometry)
        if isometry_scheduler_callable is not None
        else None
    )

    ckpt_cfg = conf.get("checkpointing", {})
    checkpoint_dir = ckpt_cfg.get("checkpoint_dir", None)
    checkpoint_every_n_steps = ckpt_cfg.get("checkpoint_every_n_steps", None)
    checkpoint_window_size = ckpt_cfg.get("checkpoint_window_size", 3)
    checkpointer = (
        Checkpointer(
            checkpoint_dir=os.path.join(checkpoint_dir, run_name),
            models={"neural_isometry": neural_isometry},
            optimizers={"isometry": optim_isometry},
            schedulers={"isometry": scheduler_isometry},
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            checkpoint_window_size=checkpoint_window_size,
            higher_is_better=False,  # we minimise the Dirichlet energy
            run_name=run_name,
            config=OmegaConf.to_container(conf, resolve=True),
        )
        if checkpoint_dir is not None
        else None
    )

    vibrational_modes(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        material=material,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        optim_isometry=optim_isometry,
        scheduler_isometry=scheduler_isometry,
        callbacks=callbacks,
        wandb_enabled=conf.wandb.enabled,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        tail_probability=conf.tail_probability,
        num_tail_samples=conf.num_tail_samples,
        model_state_kwargs=conf.get("model_state_kwargs", {}) or {},
        pushforward_kwargs=conf.get("pushforward_kwargs", {}) or {},
        checkpointer=checkpointer,
        checkpoint=checkpoint,
        max_grad_norm=conf.get("max_grad_norm", None),
        method=conf.dirichlet_energy_method,
        dirichlet_energy_kwargs=conf.get("dirichlet_energy_kwargs", {}),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
