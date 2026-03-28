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
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.regularizers import Regularizer, _base_push_cache
from infidictionary.domain_samplers import DomainSampler
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

OmegaConf.register_new_resolver("eval", eval)


def reg_change_of_basis(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    regularizers: list,           # list of (weight: float, regularizer: Regularizer)
    domain_sampler: DomainSampler,
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
    max_grad_norm: float | None = None,
    ema_decay: float | None = None,
    atom_batch_size: int | None = None,
):
    """Train the neural isometry by minimising a weighted sum of regularizer energies.

    Args:
        ema_decay:       If not None, enables adaptive weighting via an exponential moving
                         average of each term's energy magnitude.  The effective weight for
                         term i becomes ``configured_weight_i / EMA_i``, so every term
                         contributes at roughly its configured weight regardless of its raw
                         scale.  The EMA is updated once per optimizer step (from the mean
                         over gradient-accumulation steps); the previous step's estimate is
                         used during the forward pass.  Typical value: 0.99.
        atom_batch_size: If not None and the number of high-probability exact atoms exceeds
                         this value, the exact atoms are split into chunks of this size.
                         Each chunk's contribution is back-propagated immediately so that
                         only one chunk's computation graph lives in memory at a time.
                         Use this when a small ``tail_probability`` yields too many exact
                         atoms to fit in GPU memory.  Gradients accumulate correctly across
                         chunks.  When None (default) all exact atoms are processed in a
                         single forward/backward pass (original behaviour).
    """
    neural_isometry = neural_isometry.to(device)
    optim_isometry.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    # EMA state: one scale estimate per regularizer (None = not yet initialised).
    _EMA_EPS = 1e-8
    ema_scales: list[float | None] = [None] * len(regularizers)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        coords = domain_sampler.sample(domain_sample_size).to(device)
        for _, regularizer in regularizers:
            regularizer.update_coordinates(coords)

        # Effective weights for this step: use EMA-normalised weights if enabled,
        # falling back to the configured weight on the first step (EMA not yet warm).
        if ema_decay is not None:
            effective_weights = [
                w / max(s, _EMA_EPS) if s is not None else w
                for (w, _), s in zip(regularizers, ema_scales)
            ]
        else:
            effective_weights = [w for w, _ in regularizers]

        total_energy_history: list[float] = []
        per_reg_history: list[list[float]] = [[] for _ in regularizers]

        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # ── Stratified atom sampling ───────────────────────────────────────
            idx_exact = initial_dictionary.get_high_probability_indices(
                tail_probability
            ).to(device)
            pmfs_exact = initial_dictionary.get_index_pmfs(idx_exact).to(device)

            idx_all = initial_dictionary.sample_indices(num_tail_samples).to(device)
            in_exact = (
                idx_all[:, None, :] == idx_exact[None, :, :]
            ).all(dim=-1).any(dim=-1)
            idx_tail = idx_all[~in_exact]

            if idx_tail.shape[0] > 0:
                idx_tail_u, idx_tail_c = torch.unique(idx_tail, return_counts=True, dim=0)
            else:
                idx_tail_u = idx_tail_c = None

            # ── Compute total weighted energy ──────────────────────────────────
            # Split exact atoms into chunks; when atom_batch_size is None or the
            # atom count fits, this produces a single chunk (original behaviour).
            # Each chunk is back-propagated immediately so that only one chunk's
            # computation graph lives in memory at a time.
            if atom_batch_size is not None and idx_exact.shape[0] > atom_batch_size:
                exact_chunks = list(zip(
                    idx_exact.split(atom_batch_size),
                    pmfs_exact.split(atom_batch_size),
                ))
            else:
                exact_chunks = [(idx_exact, pmfs_exact)]

            # Chunks are in the outer loop so all regularizers share the cached
            # pushforward for each chunk. backward() + reset are called once per
            # chunk, keeping at most one chunk's graph in memory at a time.
            total_energy_item = 0.0
            step_per_reg = [0.0] * len(regularizers)

            for chunk_idx, chunk_pmfs in exact_chunks:
                chunk_loss = None
                for i, (_, regularizer) in enumerate(regularizers):
                    if effective_weights[i] <= 1e-12:
                        continue
                    chunk_per_atom = regularizer.compute_energy(
                        neural_isometry, initial_dictionary, coords,
                        chunk_idx, pushforward_kwargs,
                    )
                    chunk_energy = (chunk_per_atom * chunk_pmfs).sum()
                    step_per_reg[i] += chunk_energy.item()
                    term = effective_weights[i] * chunk_energy / grad_accumulation_steps
                    chunk_loss = term if chunk_loss is None else chunk_loss + term
                if chunk_loss is not None:
                    chunk_loss.backward()
                _base_push_cache.reset()

            if idx_tail_u is not None:
                tail_loss = None
                for i, (_, regularizer) in enumerate(regularizers):
                    if effective_weights[i] <= 1e-12:
                        continue
                    tail_per_atom = regularizer.compute_energy(
                        neural_isometry, initial_dictionary, coords,
                        idx_tail_u, pushforward_kwargs,
                    )
                    reg_tail = (tail_per_atom * idx_tail_c.float()).sum() / num_tail_samples
                    step_per_reg[i] += reg_tail.item()
                    term = effective_weights[i] * reg_tail / grad_accumulation_steps
                    tail_loss = term if tail_loss is None else tail_loss + term
                if tail_loss is not None:
                    tail_loss.backward()
                _base_push_cache.reset()

            for i in range(len(regularizers)):
                per_reg_history[i].append(step_per_reg[i])
                total_energy_item += effective_weights[i] * step_per_reg[i]
            total_energy_history.append(total_energy_item)

        energy_item = sum(total_energy_history) / len(total_energy_history)
        per_reg_items = [sum(h) / len(h) for h in per_reg_history]

        # ── Update EMA scales once per optimizer step ──────────────────────────
        if ema_decay is not None:
            for i, reg_item in enumerate(per_reg_items):
                abs_val = abs(reg_item)
                if ema_scales[i] is None:
                    ema_scales[i] = abs_val if abs_val > _EMA_EPS else 1.0
                else:
                    ema_scales[i] = ema_decay * ema_scales[i] + (1 - ema_decay) * abs_val

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            wandb.log({"train/total_energy": energy_item}, step=epoch_i)
            for i, ((weight, regularizer), reg_item) in enumerate(zip(regularizers, per_reg_items)):
                if effective_weights[i] <= 1e-12:
                    continue
                reg_name = type(regularizer).__name__
                wandb.log({f"{reg_name}/energy_raw": reg_item}, step=epoch_i)
                wandb.log({f"{reg_name}/energy_weighted": effective_weights[i] * reg_item}, step=epoch_i)
                if ema_decay is not None and ema_scales[i] is not None:
                    wandb.log({f"{reg_name}/ema_scale": ema_scales[i]}, step=epoch_i)
                    wandb.log({f"{reg_name}/effective_weight": effective_weights[i]}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({"energy": energy_item})
        optim_isometry.step()
        optim_isometry.zero_grad()
        step_scheduler(scheduler_isometry, energy_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=energy_item)

        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=None,
                wandb_enabled=wandb_enabled,
                device=device,
            )


@hydra.main(version_base=None, config_path="conf", config_name="reg_cob")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    domain_sampler: DomainSampler = instantiate(conf.domain_sampler)

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
            higher_is_better=False,
            run_name=run_name,
            config=OmegaConf.to_container(conf, resolve=True),
        )
        if checkpoint_dir is not None
        else None
    )

    # Build the weighted regularizer list from conf.regularizers.
    # Each entry has a 'weight' key plus the instantiation config (_target_, params…).
    # Entries with weight == 0 are skipped to avoid wasted computation.
    regularizers: list[tuple[float, Regularizer]] = []
    for entry in conf.regularizers.values():
        weight = float(entry.weight)
        if weight == 0.0:
            continue
        reg_cfg = {k: v for k, v in OmegaConf.to_container(entry, resolve=True).items() if k != "weight"}
        regularizers.append((weight, instantiate(reg_cfg)))

    aw = conf.get("adaptive_weighting", {})
    ema_decay = float(aw["ema_decay"]) if aw.get("enabled", False) else None

    reg_change_of_basis(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        regularizers=regularizers,
        domain_sampler=domain_sampler,
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
        ema_decay=ema_decay,
        atom_batch_size=conf.get("atom_batch_size", None),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
