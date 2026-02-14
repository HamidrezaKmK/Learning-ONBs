import math
from typing import Any, Callable, Literal
import matplotlib
matplotlib.use("Agg") 
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from tqdm import tqdm
import wandb

from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.index_sampler import IndexSampler
from infidictionary.regularizers import Regularizer

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)


def pwc(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    regularizer: Regularizer,
    index_sampler: IndexSampler, # defines the \ell^2 on the coefficient space (energy)
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    isometry_optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    isometry_scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    callbacks: list,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    atom_index_batch_size: int,
    reg_lambda: float = 1.0,
    fidelity_lambda: float = 1.0,
):
    neural_isometry = neural_isometry.to(device)
    optim_isometry = isometry_optimizer_callable(neural_isometry.parameters())
    scheduler_isometry = isometry_scheduler_callable(optim_isometry) if isometry_scheduler_callable is not None else None
    optim_isometry.zero_grad()

    pbar = tqdm(range(n_epochs))
    
    for epoch_i in pbar:

        if epoch_i % grad_accumulation_steps == 0:
            # TODO: add a domain sampler object
            # TODO: make this more efficient
            coords = initial_dictionary.sample_from_domain(domain_sample_size).to(device) # shape (N, d)
            fidelity_history_temp = []
            reg_energy_history_temp = []

            # use the same index samples throughout the grad accumulation steps
            # sample atom_indices as a batch of indices distributed according to a decreasing index distribution
            atom_indices = index_sampler.sample_indices(atom_index_batch_size, epoch_i, n_epochs)
            atom_indices = atom_indices.long().cpu()  # shape (A, )

            # for each function compute the inner products with all deformed atoms, thus 
            # getting a (B, A) matrix of inner products
            unique_atom_indices, atom_counts = torch.unique(atom_indices, return_counts=True)
            atom_counts = atom_counts.float().to(device)  # shape (A, )
        
        initial_atoms = initial_dictionary.get_atom(coords, unique_atom_indices.to(device))  # shape (A, N)
        transformed_bases_unique = neural_isometry.transform(
            initial_dictionary=initial_dictionary,
            atom_indices=unique_atom_indices,
            coords=coords,
            device=device,
            mode='pullback',
        ) # shape (A, N)

        all_reg = regularizer.compute_energy(
            coords,       
            transformed_bases_unique,
        ) # (A, )
        # add fidelity
        all_fidelity = torch.mean((transformed_bases_unique - initial_atoms) ** 2, dim=1)  # shape (A, )
        
        fidelity = (all_fidelity * atom_counts).sum() / atom_counts.sum()
        reg_energy = (all_reg * atom_counts).sum() / atom_counts.sum()
        energy = fidelity_lambda * fidelity + reg_lambda * reg_energy
        (energy / grad_accumulation_steps).backward(retain_graph=False)
        fidelity_history_temp.append(all_fidelity.mean().item())
        reg_energy_history_temp.append(all_reg.mean().item())

        if (epoch_i + 1) % grad_accumulation_steps == 0:
            fidelity_item = sum(fidelity_history_temp) / len(fidelity_history_temp)
            reg_energy_item = sum(reg_energy_history_temp) / len(reg_energy_history_temp)
            energy_item = fidelity_lambda * fidelity_item + reg_lambda * reg_energy_item
            
            if wandb_enabled:
                wandb.log({"train/fidelity": fidelity_item}, step=epoch_i)
                wandb.log({"train/reg_energy": reg_energy_item}, step=epoch_i)
                wandb.log({"train/total_energy": energy_item}, step=epoch_i)
                wandb.log({"train/iteration": epoch_i}, step=epoch_i)
                # visualize norm of the gradients of the parameters
                all_grads = []
                for param in neural_isometry.parameters():
                    if param.grad is not None:
                        all_grads.append(param.grad.view(-1))
                all_grads = torch.cat(all_grads)
                grad_norm = torch.norm(all_grads).item()
                wandb.log({"train/grad_norm": grad_norm}, step=epoch_i)
                # visualize the magnitude of the parameters
                all_params = []
                for param in neural_isometry.parameters():
                    all_params.append(param.view(-1))
                all_params = torch.cat(all_params)
                param_norm = torch.norm(all_params).item()
                wandb.log({"train/param_norm": param_norm}, step=epoch_i)
                avg_lr2 = sum(group['lr'] for group in optim_isometry.param_groups) / len(optim_isometry.param_groups)
                wandb.log({"train/avg_lr_isometry": avg_lr2}, step=epoch_i)

            pbar.set_postfix({'energy': energy_item})
            optim_isometry.step()
            optim_isometry.zero_grad()

            if scheduler_isometry is not None:
                if isinstance(scheduler_isometry, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_isometry.step(energy_item)
                else:
                    scheduler_isometry.step()

        
        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=None,
                wandb_enabled=wandb_enabled,
                device=device,
            )

@hydra.main(version_base=None, config_path="conf", config_name="pwc")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    index_sampler: IndexSampler = instantiate(conf.index_sampler)
    regularizer: Regularizer = instantiate(conf.regularizer)

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

    isometry_optimizer_callable = instantiate(conf.isometry_optimizer_callable)
    if conf.get("isometry_scheduler_callable", None):
        isometry_scheduler_callable = instantiate(conf.isometry_scheduler_callable)
    else:
        isometry_scheduler_callable = None
        
    pwc(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        index_sampler=index_sampler,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        isometry_optimizer_callable=isometry_optimizer_callable,
        isometry_scheduler_callable=isometry_scheduler_callable,
        callbacks=callbacks,
        wandb_enabled=conf.wandb.enabled,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        atom_index_batch_size=conf.atom_index_batch_size,
        reg_lambda=conf.reg_lambda,
        fidelity_lambda=conf.fidelity_lambda,
        regularizer=regularizer,
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
