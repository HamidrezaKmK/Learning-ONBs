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
from infidictionary.domain_samplers import DomainSampler
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)


def reg_change_of_basis(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    regularizer: Regularizer,
    domain_sampler: DomainSampler,
    n_epochs: int,
    domain_sample_size: int,
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
    reg_lambda: float = 1.0,
    fidelity_lambda: float = 1.0,
):
    neural_isometry = neural_isometry.to(device)
    optim_isometry.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        neural_isometry.shuffle_model_state(**model_state_kwargs)
        coords = domain_sampler.sample(domain_sample_size).to(device)  # (N, d)
        fidelity_history_temp = []
        reg_energy_history_temp = []

        for _ in range(grad_accumulation_steps):
            # sample atom indices from the dictionary's own prior distribution
            atom_indices = initial_dictionary.sample_indices(atom_index_batch_size).to(device)
            unique_atom_indices, inverse_indices = torch.unique(atom_indices, dim=0, return_inverse=True)
            atom_counts = torch.bincount(inverse_indices).float().to(device)  # (A_unique,)

            initial_atoms = initial_dictionary.get_atoms(coords, unique_atom_indices)  # (A, N, C)
            _, _, pushed_atoms = neural_isometry.pushforward(
                src_coords=coords,
                src_logabsdet=torch.zeros(coords.shape[0], device=device),
                src_field=initial_atoms,
                **pushforward_kwargs,
            )  # (A, N, C)

            all_reg = regularizer.compute_energy(coords, pushed_atoms)  # (A,)
            all_fidelity = torch.mean((pushed_atoms - initial_atoms) ** 2, dim=(1, 2))  # (A,)

            fidelity = (all_fidelity * atom_counts).sum() / atom_counts.sum()
            reg_energy = (all_reg * atom_counts).sum() / atom_counts.sum()
            energy = fidelity_lambda * fidelity + reg_lambda * reg_energy
            (energy / grad_accumulation_steps).backward(retain_graph=False)
            fidelity_history_temp.append(fidelity.item())
            reg_energy_history_temp.append(reg_energy.item())

        fidelity_item = sum(fidelity_history_temp) / len(fidelity_history_temp)
        reg_energy_item = sum(reg_energy_history_temp) / len(reg_energy_history_temp)
        energy_item = fidelity_lambda * fidelity_item + reg_lambda * reg_energy_item

        if wandb_enabled:
            wandb.log({"train/fidelity": fidelity_item}, step=epoch_i)
            wandb.log({"train/reg_energy": reg_energy_item}, step=epoch_i)
            wandb.log({"train/total_energy": energy_item}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({'energy': energy_item})
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


@hydra.main(version_base=None, config_path="conf", config_name="regularized_change_of_basis")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    regularizer: Regularizer = instantiate(conf.regularizer)
    domain_sampler: DomainSampler = instantiate(conf.domain_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model and optimizer state if resuming training
    checkpoint, wandb_run_id = None, None
    if conf.resume_training.enabled and conf.resume_training.checkpoint_path is not None:
        checkpoint = torch.load(conf.resume_training.checkpoint_path, weights_only=False, map_location=device)
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
    scheduler_isometry = isometry_scheduler_callable(optim_isometry) if isometry_scheduler_callable is not None else None

    # checkpoints
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
        higher_is_better=False,
        run_name=run_name,
        config=OmegaConf.to_container(conf, resolve=True),
    ) if checkpoint_dir is not None else None

    reg_change_of_basis(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        regularizer=regularizer,
        domain_sampler=domain_sampler,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
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
        reg_lambda=conf.reg_lambda,
        fidelity_lambda=conf.fidelity_lambda,
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
