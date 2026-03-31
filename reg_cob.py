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
from infidictionary.regularizers import Regularizer
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

OmegaConf.register_new_resolver("eval", eval)


def reg_change_of_basis(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    regularizer: Regularizer,
    n_epochs: int,
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
    atom_batch_size: int | None = None,
):
    """Train the neural isometry by minimising a regularizer energy.

    Args:
        atom_batch_size: If not None and the number of high-probability exact
                         atoms exceeds this value, exact atoms are split into
                         chunks of this size and each chunk is back-propagated
                         immediately, keeping only one chunk's graph in memory
                         at a time. When None all exact atoms are processed in
                         a single forward/backward pass.
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

        regularizer.update_coordinates(neural_isometry, pushforward_kwargs)

        energy_history: list[float] = []

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
            num_not_rejected = in_exact.sum().item()

            if idx_tail.shape[0] > 0:
                idx_tail_u, idx_tail_c = torch.unique(idx_tail, return_counts=True, dim=0)
            else:
                idx_tail_u = idx_tail_c = None

            # Chunking the atoms for VRAM memory limitation
            if atom_batch_size is not None and idx_exact.shape[0] > atom_batch_size:
                exact_chunks = list(zip(
                    idx_exact.split(atom_batch_size),
                    pmfs_exact.split(atom_batch_size),
                ))
            else:
                exact_chunks = [(idx_exact, pmfs_exact)]

            step_energy = 0.0
            for chunk_idx, chunk_pmfs in exact_chunks:
                chunk_per_atom = regularizer.compute_energy(
                    neural_isometry, initial_dictionary, chunk_idx, pushforward_kwargs,
                )
                chunk_loss = (chunk_per_atom * chunk_pmfs).sum()
                step_energy += chunk_loss.item()
                (chunk_loss / grad_accumulation_steps).backward()

            # ── Tail atoms ─────────────────────────────────────────────────────
            if idx_tail_u is not None:
                tail_per_atom = regularizer.compute_energy(
                    neural_isometry, initial_dictionary, idx_tail_u, pushforward_kwargs,
                )
                tail_loss = (tail_per_atom * idx_tail_c.float()).sum() / num_not_rejected
                step_energy += tail_loss.item()
                (tail_loss / grad_accumulation_steps).backward()

            energy_history.append(step_energy)

        energy_item = sum(energy_history) / len(energy_history)

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            wandb.log({"train/energy": energy_item}, step=epoch_i)
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

    regularizer: Regularizer = instantiate(conf.regularizer)

    reg_change_of_basis(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        regularizer=regularizer,
        n_epochs=conf.n_epochs,
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
        atom_batch_size=conf.get("atom_batch_size", None),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
