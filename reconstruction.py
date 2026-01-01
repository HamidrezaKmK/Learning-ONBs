import math
from typing import Any, Callable, Dict

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import wandb

from basis_learning.diffeomorphisms.base import Diffeomorphism
from basis_learning.bases.base import BaseFunction
from basis_learning.datasets import FunctionClassGenerator

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)

def train(
    diffeomorphism: Diffeomorphism,
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    callbacks: list,
    f_gen: FunctionClassGenerator,
    n_functions: int | None,
    initial_basis: BaseFunction,
    initial_basis_idx: list,
    wandb_enabled: bool,
    loss_smoothing_alpha: float = 0.99,
):
    diffeomorphism = diffeomorphism.to(device)
    optimizer = optimizer_callable(diffeomorphism.parameters())
    scheduler = scheduler_callable(optimizer) if scheduler_callable is not None else None
    

    smoothed_loss = None
    loss_history = []
    coords = initial_basis.sample_from_domain(domain_sample_size).to(device) # shape (N, d)

    pbar = tqdm(range(n_epochs))

    for epoch_i in pbar:
        if n_functions is None:
            seed = epoch_i
        else:
            seed = epoch_i % n_functions
            # TODO: bring back
            # seed = torch.randint(0, n_functions, (1,)).item()
        vals = f_gen(coords, seed=seed)  # shape (N,)

        optimizer.zero_grad()
        loss = 0
        proj = 0
        for idx_ in range(len(initial_basis_idx)):
            idx = initial_basis_idx[idx_]
            deformed_coords, logabsdet = diffeomorphism.forward(coords)
            deformed_vals = initial_basis.get(deformed_coords, idx).to(device)
            deformed_vals = deformed_vals * torch.exp(0.5 * logabsdet)
            proj += (deformed_vals * vals).mean() * deformed_vals
        loss = torch.mean((vals - proj) ** 2)
        loss.backward()
        if epoch_i % n_functions == (n_functions - 1):
            if smoothed_loss is None:
                smoothed_loss = loss.item()
            else:
                smoothed_loss = loss_smoothing_alpha * smoothed_loss + (1 - loss_smoothing_alpha) * loss.item()
            if wandb_enabled:
                wandb.log({"train/loss": smoothed_loss})
            pbar.set_postfix({'loss': smoothed_loss})
            optimizer.step()
            # check if it is reduce on plateau scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(smoothed_loss)
                else:
                    scheduler.step()
            loss_history.append(smoothed_loss)

            for callback in callbacks:
                callback(
                    epoch=epoch_i,
                    diffeomorphism=diffeomorphism,
                    loss=smoothed_loss,
                    wandb_enabled=wandb_enabled,
                    device=device,
                )

@hydra.main(version_base=None, config_path="conf", config_name="reconstruction")
def main(conf: DictConfig):

    diffeomorphism = instantiate(conf.diffeomorphism)
    function_generator = instantiate(conf.function_generator)  # dataset of datasets
    initial_basis = instantiate(conf.initial_basis)
    
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
            name=wandb_run_name
        )
        
    if "callbacks" not in conf:
        callbacks = []
    else:
        callbacks = [instantiate(callback) for callback in conf.callbacks.values()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler_callable = instantiate(conf.get("scheduler", None)) if conf.get("scheduler", None) else None
    optimizer_callable = instantiate(conf.optimizer)
    train(
        diffeomorphism=diffeomorphism,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        optimizer_callable=optimizer_callable,
        scheduler_callable=scheduler_callable,
        callbacks=callbacks,
        f_gen=function_generator,
        n_functions=conf.get("n_functions", None),
        initial_basis=initial_basis,
        initial_basis_idx=conf.initial_basis_indices,
        loss_smoothing_alpha=conf.get("loss_smoothing_alpha", 0.99),
        wandb_enabled=conf.wandb.enabled,
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()