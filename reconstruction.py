import math
from typing import Any, Callable

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import wandb

from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.datasets import FunctionClassGenerator
from infidictionary.utils import gram_projection
from infidictionary.linear_synthesis.base import OrthogonalSynthesis

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)

def train(
    diffeomorphism: Diffeomorphism,
    orthogonal_synthesis: OrthogonalSynthesis,
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    callbacks: list,
    f_gen: FunctionClassGenerator,
    n_functions: int | None,
    initial_atoms: InfiDictionary,
    wandb_enabled: bool,
    batch_size: int,
    grad_accumulation_steps: int,
    n_reduced_order: int,
    loss_smoothing_alpha: float = 0.99,
):
    diffeomorphism = diffeomorphism.to(device)
    optimizer = optimizer_callable(diffeomorphism.parameters())
    scheduler = scheduler_callable(optimizer) if scheduler_callable is not None else None

    smoothed_loss = None
    loss_history = []
    
    pbar = tqdm(range(n_epochs))    
    optimizer.zero_grad()

    losses_temp_history = []
    for epoch_i in pbar:

        if epoch_i % grad_accumulation_steps == 0:
            coords = initial_atoms.sample_from_domain(domain_sample_size).to(device) # shape (N, d)
            deformed_coords, logabsdets = diffeomorphism.forward(coords)
        
        # pick batch_size numbers from [0, n_seeds)
        n_seeds = n_functions or n_epochs
        seed_batch = torch.randperm(n_seeds)[:batch_size].to(device)
        # concatenate coords and deformed_coords accordingly
        all_coords = torch.cat([coords, deformed_coords], dim=0)  # shape (2N, d)
        all_vals = f_gen.get_batch(all_coords, seeds=seed_batch)  # shape (B, 2N)
        vals = all_vals[:, :domain_sample_size].to(device)  # shape (B, N)
        deformed_vals = all_vals[:, domain_sample_size:].to(device)  # shape (B, N)
        atom_indices = torch.arange(initial_atoms.num_atoms, device=device)
        projection = gram_projection(
            orthogonal_synthesis=orthogonal_synthesis,
            atom_indices=atom_indices,
            coords=coords,
            vals=vals,
            deformed_coords=deformed_coords,
            deformed_vals=deformed_vals,
            logabsdets=logabsdets,
            initial_dictionary=initial_atoms,
            device=device,
            n_truncation=n_reduced_order,
        ) # shape (B, N)
        loss = torch.mean((projection - vals) ** 2) / grad_accumulation_steps
        losses_temp_history.append(loss.item())
        retain = (epoch_i + 1) % grad_accumulation_steps != 0
        loss.backward(retain_graph=retain)
        if not retain:
            loss_item = sum(losses_temp_history)
            losses_temp_history = []
            if smoothed_loss is None:
                smoothed_loss = loss_item
            else:
                smoothed_loss = loss_smoothing_alpha * smoothed_loss + (1 - loss_smoothing_alpha) * loss_item
            if wandb_enabled:
                wandb.log({"train/loss": smoothed_loss})
                wandb.log({"train/iteration": epoch_i})
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
                orthogonal_synthesis=orthogonal_synthesis,
                diffeomorphism=diffeomorphism,
                wandb_enabled=wandb_enabled,
                device=device,
            )

@hydra.main(version_base=None, config_path="conf", config_name="reconstruction")
def main(conf: DictConfig):

    diffeomorphism = instantiate(conf.diffeomorphism)
    function_generator = instantiate(conf.function_generator)  # dataset of datasets
    initial_atoms: InfiDictionary = instantiate(conf.initial_atoms)
    orthogonal_synthesis_partial = instantiate(conf.orthogonal_synthesis_partial)
    orthogonal_synthesis = orthogonal_synthesis_partial(n_atoms=initial_atoms.num_atoms)
    n_reduced_order = conf.get("n_reduced_order", None) or initial_atoms.num_atoms

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
        orthogonal_synthesis=orthogonal_synthesis,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        optimizer_callable=optimizer_callable,
        scheduler_callable=scheduler_callable,
        callbacks=callbacks,
        f_gen=function_generator,
        n_functions=conf.get("n_functions", None),
        initial_atoms=initial_atoms,
        batch_size=conf.batch_size,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        loss_smoothing_alpha=conf.get("loss_smoothing_alpha", 0.99),
        wandb_enabled=conf.wandb.enabled,
        n_reduced_order=n_reduced_order,
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
