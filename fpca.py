import datetime
import math
import os
from typing import Any, Callable
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
from infidictionary.datasets import IrregularDataset
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)


def train(
    neural_isometry: NeuralIsometry,
    mean_function: NeuralField,
    f_gen: IrregularDataset,
    initial_dictionary: InfiDictionary,
    batch_size: int,
    n_epochs: int,
    device: torch.device,
    optim_isometry: torch.optim.Optimizer,
    scheduler_isometry: torch.optim.lr_scheduler._LRScheduler | None,
    optim_mean_function: torch.optim.Optimizer,
    scheduler_mean_function: torch.optim.lr_scheduler._LRScheduler | None,
    callbacks: list,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    energy_estimation_kwargs: dict,
    model_state_kwargs: dict,
    pullback_pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
):
    neural_isometry = neural_isometry.to(device)
    optim_isometry.zero_grad()

    mean_function = mean_function.to(device)
    optim_mean_function.zero_grad()

    start_epoch = 0
    best_energy = -math.inf

    if checkpoint is not None and checkpointer is not None:
        start_epoch, best_energy = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        neural_isometry.shuffle_model_state(**model_state_kwargs)
        energy_history_temp = []
        mean_function_mse_history_temp = []

        for _ in range(grad_accumulation_steps):
            coords, vals = f_gen.get_batch(batch_size)
            coords = coords.to(device)  # shape (N, d)
            vals = vals.to(device)      # shape (B, N, C)

            # (1: mean function) compute the mean function and do its backward:
            avg_vals = mean_function(coords) # shape (N, C)
            mean_mse = torch.mean((vals - avg_vals.unsqueeze(0)).pow(2))
            (mean_mse / grad_accumulation_steps).backward(retain_graph=False)
            mean_function_mse_history_temp.append(mean_mse.item())

            # (2: covariance training) zero-center the data and do KL expansion step:
            # get the vals centered and work with them
            vals_centered = (vals - avg_vals.unsqueeze(0)).detach()  # shape (B, N, C)
            src_coords, src_logabsdet, vals_pulled_back = neural_isometry.pullback(
                tgt_coords=coords,
                tgt_logabsdet=torch.zeros(coords.shape[0], device=coords.device),
                tgt_field=vals_centered,
                **pullback_pushforward_kwargs,
            )
            energy = initial_dictionary.estimate_captured_energy(
                coords=src_coords,
                logabsdet=src_logabsdet,
                values=vals_pulled_back,
                **energy_estimation_kwargs,
            ).mean()

            (-energy / grad_accumulation_steps).backward(retain_graph=False)
            energy_history_temp.append(energy.item())

        energy_item = sum(energy_history_temp) / len(energy_history_temp)
        mean_mse_item = sum(mean_function_mse_history_temp) / len(mean_function_mse_history_temp)

        if wandb_enabled:
            wandb.log({"train/energy": energy_item}, step=epoch_i)
            wandb.log({"train/mean_function_mse": mean_mse_item}, step=epoch_i)
            wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            wandb.log({"train/grad_norm_isometry": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/param_norm_isometry": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train/grad_norm_mean_function": get_grad_norm(mean_function)}, step=epoch_i)
            wandb.log({"train/param_norm_mean_function": get_param_norm(mean_function)}, step=epoch_i)
            # visualize the average learning rate of the optimizer
            wandb.log({"train/lr_mean_function": get_avg_lr(optim_mean_function)}, step=epoch_i)
            wandb.log({"train/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({'energy': energy_item, 'mean_function_mse': mean_mse_item})
        optim_isometry.step()
        optim_mean_function.step()
        optim_isometry.zero_grad()
        optim_mean_function.zero_grad()

        step_scheduler(scheduler_isometry, energy_item)
        step_scheduler(scheduler_mean_function, mean_mse_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=energy_item)

        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                wandb_enabled=wandb_enabled,
                device=device,
            )

@hydra.main(version_base=None, config_path="conf", config_name="fpca")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    function_generator: IrregularDataset = instantiate(conf.function_generator)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    mean_function: NeuralField = instantiate(conf.mean_function)
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
            # compatible with hydra
            settings=wandb.Settings(start_method="thread"),
            name=wandb_run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
        )
        run_name = f"wandb-{wandb.run.id}"
    elif wandb_run_id is not None:
        raise ValueError("You are resuming a wandb run without specifying wandb=enabled!")

    # instantiate the callbacks
    if "callbacks" not in conf:
        callbacks = []
    else:
        callbacks = [instantiate(callback) for callback in conf.callbacks.values()]

    # instantiate the optimizer and schedulers
    mean_function_optimizer_callable = instantiate(conf.mean_function_optimizer_callable)
    if conf.get("mean_function_scheduler_callable", None):
        mean_function_scheduler_callable = instantiate(conf.mean_function_scheduler_callable)
    else:
        mean_function_scheduler_callable = None

    isometry_optimizer_callable = instantiate(conf.isometry_optimizer_callable)
    if conf.get("isometry_scheduler_callable", None):
        isometry_scheduler_callable = instantiate(conf.isometry_scheduler_callable)
    else:
        isometry_scheduler_callable = None
    optim_isometry = isometry_optimizer_callable(neural_isometry.parameters())
    scheduler_isometry = isometry_scheduler_callable(optim_isometry) if isometry_scheduler_callable is not None else None

    optim_mean_function = mean_function_optimizer_callable(mean_function.parameters())
    scheduler_mean_function = mean_function_scheduler_callable(optim_mean_function) if mean_function_scheduler_callable is not None else None

    # checkpoints
    ckpt_cfg = conf.get("checkpointing", {})
    checkpoint_dir = ckpt_cfg.get("checkpoint_dir", None)
    checkpoint_dir = os.path.join(checkpoint_dir, run_name)
    checkpoint_every_n_steps = ckpt_cfg.get("checkpoint_every_n_steps", None)
    checkpoint_window_size = ckpt_cfg.get("checkpoint_window_size", 3)
    checkpointer = Checkpointer(
        checkpoint_dir=checkpoint_dir,
        models={"neural_isometry": neural_isometry, "mean_function": mean_function},
        optimizers={"isometry": optim_isometry, "mean_function": optim_mean_function},
        schedulers={"isometry": scheduler_isometry, "mean_function": scheduler_mean_function},
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        checkpoint_window_size=checkpoint_window_size,
        run_name=run_name,
        config=OmegaConf.to_container(conf, resolve=True),
    ) if checkpoint_dir is not None else None

    train(
        neural_isometry=neural_isometry,
        mean_function=mean_function,
        initial_dictionary=initial_dictionary,
        energy_estimation_kwargs=conf.get("energy_estimation_kwargs", {}) or {},
        model_state_kwargs=conf.get("model_state_kwargs", {}) or {},
        pullback_pushforward_kwargs=conf.get("pullback_pushforward_kwargs", {}) or {},
        f_gen=function_generator,
        batch_size=conf.batch_size,
        n_epochs=conf.n_epochs,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        device=device,
        optim_isometry=optim_isometry,
        scheduler_isometry=scheduler_isometry,
        optim_mean_function=optim_mean_function,
        scheduler_mean_function=scheduler_mean_function,
        callbacks=callbacks,
        wandb_enabled=conf.wandb.enabled,
        checkpointer=checkpointer,
        checkpoint=checkpoint,
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
