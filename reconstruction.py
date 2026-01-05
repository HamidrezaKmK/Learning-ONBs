import math
from typing import Any, Callable

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import wandb

from basis_learning.diffeomorphisms.base import Diffeomorphism
from basis_learning.bases.base import BaseFunction
from basis_learning.datasets import FunctionClassGenerator
from basis_learning.utils import gram_projection

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)

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
    wandb_enabled: bool,
    resample_grid_frequency: int,
    batch_size: int,
    gram_matrix_n_samples: int,
    loss_smoothing_alpha: float = 0.99,
):
    diffeomorphism = diffeomorphism.to(device)
    optimizer = optimizer_callable(diffeomorphism.parameters())
    scheduler = scheduler_callable(optimizer) if scheduler_callable is not None else None

    smoothed_loss = None
    loss_history = []
    initial_basis.compute_gram_matrix(
        n_domain_samples=gram_matrix_n_samples, 
        device=device,
    )
    
    pbar = tqdm(range(n_epochs))
    coords = initial_basis.sample_from_domain(domain_sample_size).to(device)  # shape (N, d)
    optimizer.zero_grad()
    losses_temp_history = []

    for epoch_i in pbar:
        if n_functions is None:
            seed = epoch_i
        else:
            seed = torch.randint(0, n_functions, (1,)).item()

        if (epoch_i + 1) % resample_grid_frequency == 0:
            coords = initial_basis.sample_from_domain(domain_sample_size).to(device) # shape (N, d)

        vals = f_gen(coords, seed=seed)  # shape (N,)
        projection = gram_projection(
            coords=coords,
            vals=vals,
            diffeomorphism=diffeomorphism,
            basis=initial_basis,
            device=device,
        )
        loss = torch.mean((projection - vals) ** 2) / batch_size
        losses_temp_history.append(loss.item())

        loss.backward()
        if (epoch_i + 1) % batch_size == 0:
            loss_item = sum(losses_temp_history)
            losses_temp_history = []
            if smoothed_loss is None:
                smoothed_loss = loss_item
            else:
                smoothed_loss = loss_smoothing_alpha * smoothed_loss + (1 - loss_smoothing_alpha) * loss_item
            if wandb_enabled:
                wandb.log({"train/loss": smoothed_loss})
            pbar.set_postfix({'loss': smoothed_loss})
            optimizer.step()
            optimizer.zero_grad()
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
        batch_size=conf.batch_size,
        gram_matrix_n_samples=conf.gram_matrix_n_samples,
        resample_grid_frequency=conf.resample_grid_frequency,
        loss_smoothing_alpha=conf.get("loss_smoothing_alpha", 0.99),
        wandb_enabled=conf.wandb.enabled,
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
