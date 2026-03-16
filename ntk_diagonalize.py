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
from torch.func import functional_call
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
    tail_probability: float,
    num_tail_samples: int,
    model_state_kwargs: dict,
    pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
    max_grad_norm: float | None = None,
):
    """
    Train a NeuralIsometry Q to diagonalize the NTK of a given neural field.

    At each step we maximise  Σ_a pmf_a · <Q φ_a, K Q φ_a>_{L^2}
    over the isometry parameters, using the identity

        <Q φ_a, K Q φ_a> = ‖∇_θ <f_θ, Q φ_a>‖²  =  ‖∇_θ <Q* f_θ, φ_a>‖²

    so the NTK matrix K is never explicitly formed.

    Per step:
      1.  Sample evaluation coords; evaluate f_θ (gradient tracked w.r.t. θ).
      2.  Pull back f_θ through Q* to the source domain.
      3.  For each sampled atom a:  c_a = <Q* f_θ, φ_a>_{L²(src)}.
      4.  qf_a = ‖∂c_a/∂θ‖²  (autograd.grad with create_graph=True so Q
          can be differentiated through the result).

    Atom indices are drawn via stratified sampling:
      - Exact stratum: all indices with pmf >= tail_probability, weighted by pmf.
      - MC tail stratum: num_tail_samples random indices (excluding exact),
        weighted by counts / num_tail_samples.
    """
    neural_isometry = neural_isometry.to(device)
    ntk_model = ntk_model.to(device)
    ntk_model.eval()

    optim_isometry.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        qf_history_temp = []

        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # Sample source coords; pushforward to get target eval points for f_θ
            coords_src = domain_sampler.sample(domain_sample_size).to(device)
            N = coords_src.shape[0]
            logabsdet_src = torch.zeros(N, device=device)
            with torch.no_grad():
                coords_tgt, logabsdet_tgt, _ = neural_isometry.pushforward(
                    src_coords=coords_src,
                    src_logabsdet=logabsdet_src,
                    src_field=torch.zeros(1, N, 1, device=device),
                    **pushforward_kwargs,
                )

            # Cache param names once for functional_call inside _qf_for_atoms
            ntk_param_names = [name for name, _ in ntk_model.named_parameters()]

            def _qf_for_atoms(unique_indices: torch.Tensor) -> torch.Tensor:
                """Compute <Qφ_a, K Qφ_a> = ‖∇_θ <Q*f_θ, φ_a>‖² per atom.

                Uses torch.autograd.functional.jacobian with vectorize=True to batch
                all A backward passes via vmap — no Python for-loop over atoms.

                Returns qf of shape (A,).
                """
                A = unique_indices.shape[0]
                phi_src = initial_dictionary.get_atoms(coords_src, unique_indices)  # (A, N, C)

                def c_from_ntk_params(*params):
                    ntk_dict = dict(zip(ntk_param_names, params))
                    f_vals = functional_call(ntk_model, ntk_dict, (coords_tgt,))
                    _, _, qsf = neural_isometry.pullback(
                        tgt_coords=coords_tgt,
                        tgt_logabsdet=logabsdet_tgt,
                        tgt_field=f_vals.unsqueeze(0),
                        **pushforward_kwargs,
                    )
                    # Inner product in the source domain with source measure
                    return pairwise_inner_product(
                        qsf.squeeze(0).unsqueeze(0), phi_src, logabsdet_src,
                    ).squeeze(0)  # (A,)

                # J_tuple[i] has shape (A, *param_i.shape); vectorize=True batches
                # the A backward passes via vmap instead of a Python loop.
                J_tuple = torch.autograd.functional.jacobian(
                    c_from_ntk_params, tuple(ntk_model.parameters()),
                    create_graph=True, vectorize=True,
                )
                return sum(j.reshape(A, -1).pow(2).sum(-1) for j in J_tuple)  # (A,)

            # ── Exact high-probability stratum ────────────────────────────────
            idx_exact = initial_dictionary.get_high_probability_indices(
                tail_probability
            ).to(device)
            qf_exact_per_atom = _qf_for_atoms(idx_exact)  # (A_exact,)
            pmfs_exact = initial_dictionary.get_index_pmfs(idx_exact).to(device)
            qf_exact = (qf_exact_per_atom * pmfs_exact).sum()

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
                qf_tail_per_atom = _qf_for_atoms(idx_tail_u)  # (A_tail,)
                qf_tail = (
                    qf_tail_per_atom * idx_tail_c.float()
                ).sum() / num_tail_samples
            else:
                qf_tail = torch.zeros(1, device=device).squeeze()

            total_qf = qf_exact + qf_tail

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
        tail_probability=conf.tail_probability,
        num_tail_samples=conf.num_tail_samples,
        model_state_kwargs=conf.get("model_state_kwargs", {}) or {},
        pushforward_kwargs=conf.get("pushforward_kwargs", {}) or {},
        checkpointer=checkpointer,
        checkpoint=checkpoint,
        max_grad_norm=conf.get("max_grad_norm", None),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
