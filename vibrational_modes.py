"""vibrational_modes.py

Train a NeuralIsometry Q to find the vibrational modes of a material.

Given a material with spatially varying diffusivity D(x), the vibrational modes
are the eigenfunctions of the weighted Laplacian  -∇·(D(x) ∇).  The k-th mode
minimises the weighted Dirichlet energy

    E_k[f] = ∫ D(x) ‖∇f(x)‖² dx

subject to f being L²-orthogonal to the first k-1 modes.

We find all modes simultaneously by training Q to MAXIMISE the expected
heat-kernel quadratic form over atoms drawn from the prior:

    L(Q) = E_{i ~ pmf} [ ⟨Qφ_i, P_t Qφ_i⟩_{L²} ]

where P_t = e^{t ∇·(D(x)∇)} is the heat semigroup of the weighted Laplacian
and φ_i are the Fourier basis atoms on the source domain.

Key properties of this objective:
  - P_t is compact, positive, self-adjoint with eigenvalues e^{-t λ_k} ∈ (0,1].
  - The constant function (λ_0 = 0) has the LARGEST eigenvalue (= 1).
  - Maximising ⟨f, P_t f⟩ is equivalent to minimising the Dirichlet energy:
        ⟨f, P_t f⟩ = Σ_k |⟨f, e_k⟩|² e^{-t λ_k}
    which is maximised by aligning f with the lowest-energy eigenfunctions.
  - The objective is bounded in (0, 1] and naturally normalised — no
    per-atom energy rescaling required.

P_t f(x) is approximated probabilistically via a single Euler–Maruyama step
of the underlying diffusion dX = ∇D dt + √(2D) dW:

    P_t f(x) ≈ E_Z[ f(x + √(2 D(x) t) Z) ],   Z ~ N(0, I_d)

with periodic wrapping into [0, 1)^d.

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


def heat_kernel_qform_for_atoms(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    tgt_coords: torch.Tensor,       # (N, d) detached
    diffusivity: torch.Tensor,      # (N,) detached
    mass: torch.Tensor,             # (N,) detached — ρ(x), from material
    unique_indices: torch.Tensor,   # (A, domain_dim+1)
    pushforward_kwargs: dict,
    smoothing_t: float = 1.0,
    n_diffusion_samples: int = 4,
    **kwargs,
) -> torch.Tensor:                  # (A,)
    """Compute ⟨Qφ_a, P̃_t Qφ_a⟩_ρ for each atom via Monte Carlo diffusion.

    With mass ρ the effective operator is ρ⁻¹L (L = -∇·(D∇)), whose heat
    semigroup P̃_t = e^{-t ρ⁻¹ L} corresponds to the SDE (drift dropped):

        dX ≈ √(2 D(x) / ρ(x)) dW

    The quadratic form is:

        ⟨Qφ_a, P̃_t Qφ_a⟩_ρ
            = E_x[ρ(x) · Qφ_a(x) · E_Z[Qφ_a(x + √(2D(x)t/ρ(x)) Z)]]

    where Z ~ N(0, I_d) and coordinates are wrapped periodically into [0,1)^d.

    Args:
        neural_isometry:      The isometry Q.
        initial_dictionary:   Fourier atom dictionary.
        tgt_coords:           Spatial quadrature points, shape (N, d).
        diffusivity:          D(x_j) at each point, shape (N,).
        mass:                 ρ(x_j) at each point, shape (N,).  Returned by
                              the material alongside diffusivity.
        unique_indices:       Atom multi-indices, shape (A, domain_dim+1).
        pushforward_kwargs:   Forwarded to pullback / pushforward.
        smoothing_t:          Diffusion time t in P̃_t = e^{-t ρ⁻¹ L}.
        n_diffusion_samples:  MC draws K to estimate E_Z[Qφ(x + √(2Dt/ρ)Z)].

    Returns:
        qform: shape (A,), the quadratic form ⟨Qφ_a, P̃_t Qφ_a⟩_ρ per atom.
    """
    tgt_coords = tgt_coords.detach()
    A = unique_indices.shape[0]
    N, d = tgt_coords.shape
    device = tgt_coords.device
    dtype = tgt_coords.dtype

    def _eval_qphi(coords_: torch.Tensor) -> torch.Tensor:
        """Evaluate Qφ at arbitrary coordinates, shape (A, N_, C)."""
        N_ = coords_.shape[0]
        zeros_ = torch.zeros(N_, device=device, dtype=dtype)
        if isinstance(neural_isometry, EulerianIsometry):
            src_, logdet_ = coords_, zeros_
        else:
            dummy_field = torch.zeros(1, N_, 1, device=device, dtype=dtype)
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
        )
        return qphi_  # (A, N_, C)

    # ── Qφ(x) at the original quadrature points ───────────────────────────────
    qphi_base = _eval_qphi(tgt_coords)  # (A, N, C)

    # ── MC estimate of P̃_t Qφ(x) = E_Z[ Qφ(x + √(2D(x)t/ρ(x)) Z) ] ─────────
    # Local diffusion std: √(2 D(x_j) t / ρ(x_j)), shape (N, 1)
    diffusion_std = torch.sqrt(2.0 * diffusivity * smoothing_t / mass).unsqueeze(-1)

    qphi_diffused = torch.zeros_like(qphi_base)
    for _ in range(n_diffusion_samples):
        Z = torch.randn(N, d, device=device, dtype=dtype)
        coords_shifted = tgt_coords + diffusion_std * Z        # (N, d)
        coords_shifted = coords_shifted - torch.floor(coords_shifted)  # periodic
        qphi_diffused = qphi_diffused + _eval_qphi(coords_shifted)

    qphi_diffused = qphi_diffused / n_diffusion_samples  # (A, N, C) — P̃_t Qφ

    # ── ⟨Qφ, P̃_t Qφ⟩_ρ with ρ-weighted spatial measure ─────────────────────
    log_mass = torch.log(mass)  # (N,)
    return parallel_inner_product(qphi_base, qphi_diffused, logabsdet=log_mass)


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
    smoothing_t: float,
    n_diffusion_samples: int,
    max_grad_norm: float | None = None,
):
    """Train Q to maximise E_{i~pmf}[⟨Qφ_i, P_t Qφ_i⟩].

    Per step:
      1. Sample target-domain coords from material.domain_sampler.
      2. Evaluate D(x) (detached — no gradient through the material).
      3. Draw atom indices using stratified sampling (exact + MC tail).
      4. Compute ⟨Qφ_a, P_t Qφ_a⟩ per atom via MC diffusion.
      5. Accumulate the pmf-weighted expected quadratic form and back-propagate
         its negative (gradient ascent via a descent optimiser).
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

        qform_history_temp = []

        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # ── Sample target-domain quadrature points ────────────────────────
            coords_raw = material.domain_sampler.sample(domain_sample_size).to(device)

            with torch.no_grad():
                diffusivity, mass = material(coords_raw)
                diffusivity = diffusivity.detach()  # (N,)
                mass = mass.detach()                # (N,)

            tgt_coords = coords_raw.detach()

            # ── Exact high-probability stratum ────────────────────────────────
            idx_exact = initial_dictionary.get_high_probability_indices(
                tail_probability
            ).to(device)
            qform_exact_per_atom = heat_kernel_qform_for_atoms(
                neural_isometry, initial_dictionary,
                tgt_coords, diffusivity, mass,
                idx_exact, pushforward_kwargs,
                smoothing_t=smoothing_t,
                n_diffusion_samples=n_diffusion_samples,
            )  # (A_exact,)
            pmfs_exact = initial_dictionary.get_index_pmfs(idx_exact).to(device)
            qform_exact = (qform_exact_per_atom * pmfs_exact).sum()

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
                qform_tail_per_atom = heat_kernel_qform_for_atoms(
                    neural_isometry, initial_dictionary,
                    tgt_coords, diffusivity, mass,
                    idx_tail_u, pushforward_kwargs,
                    smoothing_t=smoothing_t,
                    n_diffusion_samples=n_diffusion_samples,
                )  # (A_tail,)
                qform_tail = (
                    qform_tail_per_atom * idx_tail_c.float()
                ).sum() / num_tail_samples
            else:
                qform_tail = torch.zeros(1, device=device).squeeze()

            total_qform = qform_exact + qform_tail
            # Gradient ASCENT: minimise the negative quadratic form.
            (-total_qform / grad_accumulation_steps).backward(retain_graph=False)
            qform_history_temp.append(total_qform.item())

        qform_item = sum(qform_history_temp) / len(qform_history_temp)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            wandb.log({"train/heat_kernel_qform": qform_item}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({"heat_kernel_qform": qform_item})
        optim_isometry.step()
        optim_isometry.zero_grad()
        step_scheduler(scheduler_isometry, qform_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=qform_item)

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
            higher_is_better=True,  # we maximise the heat-kernel quadratic form
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
        smoothing_t=conf.smoothing_t,
        n_diffusion_samples=conf.n_diffusion_samples,
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
