import math
from typing import Any, Callable
import matplotlib
matplotlib.use("Agg") 
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm
import wandb

from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.datasets import FunctionClassGenerator
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.index_sampler import IndexSampler

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)

def train(
    neural_isometry: NeuralIsometry,
    f_gen: FunctionClassGenerator,
    initial_atoms: InfiDictionary,
    index_sampler: IndexSampler, # defines the \ell^2 on the coefficient space (energy)
    batch_size: int,
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    callbacks: list,
    n_functions: int | None,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    atom_index_batch_size: int,
    energy_smoothing_alpha: float = 0.99,
):
    # create optimizer on both diffeomorphism and orthogonal_synthesis parameters
    neural_isometry = neural_isometry.to(device)
    optimizer = optimizer_callable(neural_isometry.parameters())
    scheduler = scheduler_callable(optimizer) if scheduler_callable is not None else None

    smoothed_energy = None
    energies_history = []
    
    pbar = tqdm(range(n_epochs))    
    optimizer.zero_grad()

    energies_temp_history = []
    for epoch_i in pbar:

        if epoch_i % grad_accumulation_steps == 0:
            # TODO: add a domain sampler object
            # TODO: make this more efficient
            coords = initial_atoms.sample_from_domain(domain_sample_size).to(device) # shape (N, d)
        
        # pick batch_size numbers from [0, n_seeds)
        n_seeds = n_functions or n_epochs
        seed_batch = torch.randperm(n_seeds)[:batch_size].to(device)
        vals = f_gen.get_batch(coords, seeds=seed_batch).to(device)  # shape (B, N)
        
        # sample atom_indices as a batch of indices distributed according to a decreasing index distribution
        atom_indices = index_sampler.sample_indices(atom_index_batch_size, epoch_i, n_epochs)
        atom_indices = atom_indices.long().to(device)  # shape (A, )

        # for each function compute the inner products with all deformed atoms, thus 
        # getting a (B, A) matrix of inner products
        b = neural_isometry.inner_products( 
            atom_indices=atom_indices,
            coords=coords,
            vals=vals,
            initial_dictionary=initial_atoms,
            device=device,
        )
        # TODO: figure out why the captured energy scales when changing the distribution
        # due to isometry, the initial dictionary gram projection is used to compute the coefficients
        coeffs = initial_atoms.gram_solve(atom_indices, b) 

        # maximize the captured energy
        captured_energy = torch.mean(coeffs ** 2)
        loss = - captured_energy / grad_accumulation_steps
        energies_temp_history.append(captured_energy)

        loss.backward(retain_graph=False)

        if (epoch_i + 1) % grad_accumulation_steps == 0:
            energy_item = sum(energies_temp_history) / len(energies_temp_history)
            energies_temp_history = []
            if smoothed_energy is None:
                smoothed_energy = energy_item
            else:
                smoothed_energy = energy_smoothing_alpha * smoothed_energy + (1 - energy_smoothing_alpha) * energy_item
            if wandb_enabled:
                wandb.log({"train/captured_energy": smoothed_energy}, step=epoch_i)
                wandb.log({"train/iteration": epoch_i}, step=epoch_i)
            pbar.set_postfix({'captured_energy': smoothed_energy})
            optimizer.step()
            optimizer.zero_grad()
            # check if it is reduce on plateau scheduler
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(smoothed_energy)
                else:
                    scheduler.step()
            energies_history.append(smoothed_energy)
        
        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                wandb_enabled=wandb_enabled,
                device=device,
            )

@hydra.main(version_base=None, config_path="conf", config_name="reconstruction")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    function_generator: FunctionClassGenerator = instantiate(conf.function_generator)  # dataset of datasets
    initial_atoms: InfiDictionary = instantiate(conf.initial_atoms)
    index_sampler: IndexSampler = instantiate(conf.index_sampler)

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
        neural_isometry=neural_isometry,
        f_gen=function_generator,
        initial_atoms=initial_atoms,
        index_sampler=index_sampler,
        batch_size=conf.batch_size,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        optimizer_callable=optimizer_callable,
        scheduler_callable=scheduler_callable,
        callbacks=callbacks,
        n_functions=conf.get("n_functions", None),
        wandb_enabled=conf.wandb.enabled,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        atom_index_batch_size=conf.atom_index_batch_size,
        energy_smoothing_alpha=conf.get("energy_smoothing_alpha", 0.99),
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
