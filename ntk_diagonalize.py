import datetime
import math
import os

import matplotlib
matplotlib.use("Agg")
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from infidictionary.ntk import estimate_ntk
from infidictionary.checkpointing import Checkpointer
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.utils import NeuralField, pairwise_inner_product, parallel_inner_product
from infidictionary.domain_samplers import DomainSampler
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


def ntk_diagonalize(
    neural_isometry: NeuralIsometry,
    ntk_model: NeuralField,
    initial_dictionary: InfiDictionary,
    domain_sampler: DomainSampler,
    n_epochs: int,
    domain_sample_size: int,
    ntk_n_samples: int,
    ntk_sigma: float,
    ntk_reestimate_every: int,
    device: torch.device,
    optim_isometry: torch.optim.Optimizer,
    scheduler_isometry: torch.optim.lr_scheduler._LRScheduler | None,
    callbacks: list,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    atom_index_batch_size: int,
    model_state_kwargs: dict,
    pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
    max_grad_norm: float | None = None,
    tail_probability: float | None = None,
    num_tail_samples: int | None = None,
):
    """
    Train a NeuralIsometry Q to diagonalize the NTK of a given neural field.

    At each step we maximise  sum_k  <Q phi_k, K Q phi_k>_{L^2}
    over the isometry parameters, where phi_k are atoms drawn from the
    initial dictionary and K is the NTK operator.

    The quadratic form is computed with full measure-correctness:

      1.  Pull back every NTK row K(x_i, .) (viewed as a function in L^2(tgt))
          to the source domain via Q^*, capturing
              (coords_src, logabsdet_src, K_rows_src).
          The volume correction logabsdet_src accounts for the measure change
          under the pullback.

      2.  Evaluate initial atoms phi_a at coords_src.

      3.  Compute g_a(x_i) = <Q^* K(x_i, .), phi_a>_{L^2(src, logabsdet_src)}
          for all (i, a) simultaneously via pairwise_inner_product. By the
          adjoint identity this equals (K Q phi_a)(x_i), the NTK operator
          applied to Q phi_a, evaluated at target point x_i.

      4.  Push phi_a forward to the target domain, capturing the output
          measure (tgt_logabsdet) alongside Qphi_a_tgt.

      5.  Compute QF_a = <Q phi_a, g_a>_{L^2(tgt, tgt_logabsdet)}.
          The tgt_logabsdet from step 4 is passed to parallel_inner_product
          to correctly weight the integration in the target domain.

    The NTK is evaluated at fixed target coordinates (sampled at start-up)
    and is optionally refreshed together with the coords every
    `ntk_reestimate_every` epochs (0 means estimate once, never refresh).

    Notes
    -----
    * For Eulerian isometries both logabsdet_src and tgt_logabsdet are zero
      and all coordinates remain unchanged, so steps 1–5 reduce to simple
      inner products at the fixed sample points.
    * For Lagrangian / mixed isometries the measure corrections are non-trivial
      and essential for correctness.
    * The NTK model is never trained; its parameters are resampled inside
      estimate_ntk and the model is kept in eval mode throughout.
    * When tail_probability and num_tail_samples are both set, atom indices
      are drawn via stratified sampling (mirroring monte_carlo_captured_energy):
        - Exact stratum: all indices with pmf >= tail_probability, weighted by pmf.
        - MC tail stratum: num_tail_samples random indices (excluding exact),
          weighted by counts / num_tail_samples.
      Otherwise falls back to plain MC with atom_index_batch_size samples.
    """
    neural_isometry = neural_isometry.to(device)
    ntk_model = ntk_model.to(device)
    ntk_model.eval()

    optim_isometry.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    # ── Initial NTK estimation ────────────────────────────────────────────────
    coords_tgt = domain_sampler.sample(domain_sample_size).to(device)  # (N, d)
    tqdm.write(
        f"Estimating NTK: {ntk_n_samples} samples, sigma={ntk_sigma}, "
        f"N={coords_tgt.shape[0]} points ..."
    )
    with torch.no_grad():
        K_flat = estimate_ntk(ntk_model, coords_tgt, ntk_n_samples, ntk_sigma)
    N = coords_tgt.shape[0]
    C = K_flat.shape[0] // N  # output channels
    tqdm.write(
        f"NTK done. K_flat: {K_flat.shape},  "
        f"||K||_F = {K_flat.norm().item():.3e}"
    )

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        # ── Optionally re-estimate NTK on fresh domain coords ────────────────
        if ntk_reestimate_every > 0 and epoch_i > 0 and epoch_i % ntk_reestimate_every == 0:
            coords_tgt = domain_sampler.sample(domain_sample_size).to(device)
            with torch.no_grad():
                K_flat = estimate_ntk(ntk_model, coords_tgt, ntk_n_samples, ntk_sigma)
            N = coords_tgt.shape[0]
            C = K_flat.shape[0] // N

        qf_history_temp = []

        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # ── Step 1: Pull back all NTK rows to the source domain ───────────
            #
            # Reshape K_flat so that each row becomes one C-channel function
            # at the N target points:
            #   K_rows_tgt[i*C+c, j, c'] = K_flat[i*C+c, j*C+c']
            #                             = K_{c,c'}(x_i, x_j)
            K_rows_tgt = K_flat.reshape(N * C, N, C)  # (N*C, N, C)

            coords_src, logabsdet_src, K_rows_src = neural_isometry.pullback(
                tgt_coords=coords_tgt,
                tgt_logabsdet=torch.zeros(N, device=device),
                tgt_field=K_rows_tgt,
                **pushforward_kwargs,
            )
            # coords_src:    (N, d)       — source coordinates
            # logabsdet_src: (N,)         — log |det dQ^{-1}| at each source point
            # K_rows_src:    (N*C, N, C)  — pulled-back kernel rows Q^* K(x_i, .)

            def _qf_for_atoms(unique_indices: torch.Tensor) -> torch.Tensor:
                """Run steps 2–5 for a given set of unique atom indices.

                Returns qf of shape (A,).
                """
                A = unique_indices.shape[0]

                # Step 2: atoms at source coords
                phi_src = initial_dictionary.get_atoms(coords_src, unique_indices)  # (A, N, C)

                # Step 3: g_a(x_i) = <Q* K(x_i,.), phi_a>_{L^2(src)}
                g_raw = pairwise_inner_product(K_rows_src, phi_src, logabsdet_src)  # (N*C, A)
                g = g_raw.reshape(N, C, A).permute(2, 0, 1).contiguous()  # (A, N, C)

                # Step 4: push atoms forward, capture target measure
                _, tgt_logabsdet, Qphi_tgt = neural_isometry.pushforward(
                    src_coords=coords_src,
                    src_logabsdet=logabsdet_src,
                    src_field=phi_src,
                    **pushforward_kwargs,
                )  # tgt_logabsdet: (N,),  Qphi_tgt: (A, N, C)

                # Step 5: QF_a = <Q phi_a, g_a>_{L^2(tgt)}
                return parallel_inner_product(Qphi_tgt, g, tgt_logabsdet)  # (A,)

            use_stratified = (
                tail_probability is not None and num_tail_samples is not None
            )

            if use_stratified:
                # ── Exact high-probability stratum ────────────────────────────
                idx_exact = initial_dictionary.get_high_probability_indices(
                    tail_probability
                ).to(device)
                qf_exact_per_atom = _qf_for_atoms(idx_exact)  # (A_exact,)
                pmfs_exact = initial_dictionary.get_index_pmfs(idx_exact).to(device)
                qf_exact = (qf_exact_per_atom * pmfs_exact).sum()

                # ── MC tail stratum ───────────────────────────────────────────
                idx_all = initial_dictionary.sample_indices(num_tail_samples).to(device)
                in_exact = (
                    idx_all[:, None, :] == idx_exact[None, :, :]
                ).all(dim=-1).any(dim=-1)
                idx_tail = idx_all[~in_exact]

                if idx_tail.shape[0] > 0:
                    idx_tail_u, idx_tail_c = torch.unique(
                        idx_tail, return_counts=True, dim=0
                    )
                    qf_tail_per_atom = _qf_for_atoms(idx_tail_u)  # (A_tail,)
                    qf_tail = (
                        qf_tail_per_atom * idx_tail_c.float()
                    ).sum() / num_tail_samples
                else:
                    qf_tail = torch.zeros(1, device=device).squeeze()

                total_qf = qf_exact + qf_tail
            else:
                # ── Plain MC fallback ─────────────────────────────────────────
                atom_indices = initial_dictionary.sample_indices(
                    atom_index_batch_size
                ).to(device)
                unique_atom_indices, inverse_indices = torch.unique(
                    atom_indices, dim=0, return_inverse=True
                )
                atom_counts = torch.bincount(inverse_indices).float().to(device)

                qf = _qf_for_atoms(unique_atom_indices)  # (A_unique,)
                total_qf = (qf * atom_counts).sum() / atom_counts.sum()

            (-total_qf / grad_accumulation_steps).backward(retain_graph=False)
            qf_history_temp.append(total_qf.item())

        qf_item = sum(qf_history_temp) / len(qf_history_temp)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            wandb.log({"train/ntk_quadratic_form": qf_item}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({"ntk_qf": qf_item})
        optim_isometry.step()
        optim_isometry.zero_grad()
        step_scheduler(scheduler_isometry, qf_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=qf_item)

        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=None,
                wandb_enabled=wandb_enabled,
                device=device,
            )


@hydra.main(version_base=None, config_path="conf", config_name="ntk_diagonalize")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    ntk_model: NeuralField = instantiate(conf.ntk_model)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    domain_sampler: DomainSampler = instantiate(conf.domain_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally load pre-trained weights into the NTK model.
    # The architecture is still defined by conf.ntk_model; only the weights are
    # replaced.  estimate_ntk will then centre its perturbations on those weights
    # (sigma=0 => exact empirical NTK at the trained parameters).
    ntk_weights_path = conf.get("ntk_model_weights_path", None)
    if ntk_weights_path is not None:
        # Resolve relative to the original working directory (Hydra changes cwd)
        from hydra.utils import get_original_cwd
        ntk_weights_path = os.path.join(get_original_cwd(), ntk_weights_path)
        ckpt = torch.load(ntk_weights_path, weights_only=False, map_location=device)
        ntk_model.load_state_dict(ckpt["model_state_dict"])
        tqdm.write(f"Loaded NTK model weights from: {ntk_weights_path}")

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

    if "callbacks" not in conf:
        callbacks = []
    else:
        callbacks = [instantiate(callback) for callback in conf.callbacks.values()]

    isometry_optimizer_callable = instantiate(conf.isometry_optimizer_callable)
    if conf.get("isometry_scheduler_callable", None):
        isometry_scheduler_callable = instantiate(conf.isometry_scheduler_callable)
    else:
        isometry_scheduler_callable = None

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
    checkpointer = Checkpointer(
        checkpoint_dir=os.path.join(checkpoint_dir, run_name),
        models={"neural_isometry": neural_isometry},
        optimizers={"isometry": optim_isometry},
        schedulers={"isometry": scheduler_isometry},
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        checkpoint_window_size=checkpoint_window_size,
        higher_is_better=True,  # we maximise the NTK quadratic form
        run_name=run_name,
        config=OmegaConf.to_container(conf, resolve=True),
    ) if checkpoint_dir is not None else None

    ntk_diagonalize(
        neural_isometry=neural_isometry,
        ntk_model=ntk_model,
        initial_dictionary=initial_dictionary,
        domain_sampler=domain_sampler,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        ntk_n_samples=conf.ntk_n_samples,
        ntk_sigma=conf.ntk_sigma,
        ntk_reestimate_every=conf.get("ntk_reestimate_every", 0),
        device=device,
        optim_isometry=optim_isometry,
        scheduler_isometry=scheduler_isometry,
        callbacks=callbacks,
        wandb_enabled=conf.wandb.enabled,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        atom_index_batch_size=conf.atom_index_batch_size,
        model_state_kwargs=conf.get("model_state_kwargs", {}) or {},
        pushforward_kwargs=conf.get("pushforward_kwargs", {}) or {},
        checkpointer=checkpointer,
        checkpoint=checkpoint,
        max_grad_norm=conf.get("max_grad_norm", None),
        tail_probability=conf.get("tail_probability", None),
        num_tail_samples=conf.get("num_tail_samples", None),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
