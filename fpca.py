import datetime
import os

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
from infidictionary.networks import NeuralField



def get_grad_norm(module: torch.nn.Module) -> float:
    """Global L2 norm of a module's gradients."""
    grads = [p.grad.view(-1) for p in module.parameters() if p.grad is not None]
    return torch.norm(torch.cat(grads)).item() if grads else 0.0


def get_param_norm(module: torch.nn.Module) -> float:
    """Global L2 norm of a module's parameters."""
    params = [p.view(-1) for p in module.parameters()]
    return torch.norm(torch.cat(params)).item() if params else 0.0


def get_avg_lr(optimizer: torch.optim.Optimizer) -> float:
    return sum(g["lr"] for g in optimizer.param_groups) / len(optimizer.param_groups)


def step_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler | None, metric: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()


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
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    n_epochs_mean_function: int | None,
    energy_estimation_kwargs: dict,
    model_state_kwargs: dict,
    pullback_pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
    max_grad_norm: float | None = None,
):
    neural_isometry = neural_isometry.to(device)
    mean_function = mean_function.to(device)
    # Dictionaries such as Fourier/Haar/Voronoi can own a learned synthesis
    # module; leaving it on CPU here restores its momentum buffers on CPU and
    # later conflicts with CUDA parameters after atom evaluation.
    initial_dictionary = initial_dictionary.to(device)

    optim_isometry.zero_grad()
    optim_mean_function.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue

        energy_history_temp = []
        mean_function_mse_history_temp = []
        mean_function_frozen = n_epochs_mean_function is not None and epoch_i >= n_epochs_mean_function
        f_gen.reset_buffer()
        for _ in range(grad_accumulation_steps):
            neural_isometry.shuffle_model_state(**model_state_kwargs)
            coords, vals = f_gen.get_batch(batch_size)
            coords = coords.to(device)  # shape (N, d)
            vals = vals.to(device)      # shape (B, N, C)

            # (1: mean function) compute the mean function and do its backward:
            avg_vals = mean_function(coords) # shape (N, C)
            mean_mse = torch.mean((vals - avg_vals.unsqueeze(0)).pow(2))
            if not mean_function_frozen:
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

            energy = initial_dictionary.monte_carlo_captured_energy(
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
            wandb.log({"train_stats/grad_norm_isometry": get_grad_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train_stats/param_norm_isometry": get_param_norm(neural_isometry)}, step=epoch_i)
            wandb.log({"train_stats/grad_norm_mean_function": get_grad_norm(mean_function)}, step=epoch_i)
            wandb.log({"train_stats/param_norm_mean_function": get_param_norm(mean_function)}, step=epoch_i)
            wandb.log({"train_stats/lr_mean_function": get_avg_lr(optim_mean_function)}, step=epoch_i)
            wandb.log({"train_stats/avg_lr_isometry": get_avg_lr(optim_isometry)}, step=epoch_i)

        pbar.set_postfix({'energy': energy_item, 'mean_function_mse': mean_mse_item})
        if max_grad_norm is not None:
            # The isometry optimizer also steps the dictionary's synthesis block.
            isometry_params = [p for group in optim_isometry.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(isometry_params, max_grad_norm)
        optim_isometry.step()
        if not mean_function_frozen:
            optim_mean_function.step()
        optim_isometry.zero_grad()
        optim_mean_function.zero_grad()

        step_scheduler(scheduler_isometry, energy_item)
        if not mean_function_frozen:
            step_scheduler(scheduler_mean_function, mean_mse_item)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=energy_item)

@hydra.main(version_base=None, config_path="conf", config_name="fpca")
def main(conf: DictConfig):
    torch.manual_seed(conf.seed)

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    function_generator: IrregularDataset = instantiate(conf.function_generator)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    mean_function: NeuralField = instantiate(conf.mean_function)

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
            name=wandb_run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
        )
        run_name = f"wandb-{wandb.run.id}"
    elif wandb_run_id is not None:
        raise ValueError("You are resuming a wandb run without specifying wandb=enabled!")

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
    isometry_and_dictionary_parameters = [
        *neural_isometry.parameters(),
        *initial_dictionary.parameters(),
    ]
    optim_isometry = isometry_optimizer_callable(isometry_and_dictionary_parameters)
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
        models={
            "neural_isometry": neural_isometry,
            "mean_function": mean_function,
            "initial_dictionary": initial_dictionary,
        },
        optimizers={"isometry": optim_isometry, "mean_function": optim_mean_function},
        schedulers={"isometry": scheduler_isometry, "mean_function": scheduler_mean_function},
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        checkpoint_window_size=checkpoint_window_size,
        run_name=run_name,
        config=OmegaConf.to_container(conf, resolve=True),
    ) if checkpoint_dir is not None else None

    if checkpoint is not None and "initial_dictionary" not in checkpoint.get("models", {}):
        raise ValueError(
            "This checkpoint predates dictionary-synthesis checkpointing and cannot "
            "be resumed. Start a new FPCA run."
        )

    try:
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
            n_epochs_mean_function=conf.get("n_epochs_mean_function", None),
            grad_accumulation_steps=conf.grad_accumulation_steps,
            device=device,
            optim_isometry=optim_isometry,
            scheduler_isometry=scheduler_isometry,
            optim_mean_function=optim_mean_function,
            scheduler_mean_function=scheduler_mean_function,
            wandb_enabled=conf.wandb.enabled,
            checkpointer=checkpointer,
            checkpoint=checkpoint,
            max_grad_norm=conf.get("max_grad_norm", None),
        )
    finally:
        if conf.wandb.enabled:
            wandb.finish()

if __name__ == "__main__":
    main()
