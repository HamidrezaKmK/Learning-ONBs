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
from infidictionary.utils import NeuralField

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)

# TODO: add a mean function learner (translation is not implemented here)
def train(
    neural_isometry: NeuralIsometry,
    mean_function: NeuralField,
    f_gen: FunctionClassGenerator,
    initial_dictionary: InfiDictionary,
    index_sampler: IndexSampler, # defines the \ell^2 on the coefficient space (energy)
    batch_size: int,
    n_epochs: int,
    domain_sample_size: int,
    device: torch.device,
    mean_function_optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    mean_function_scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    isometry_optimizer_callable: Callable[[Any,], torch.optim.Optimizer],
    isometry_scheduler_callable: Callable[[torch.optim.Optimizer,], torch.optim.lr_scheduler._LRScheduler],
    callbacks: list,
    n_functions: int | None,
    wandb_enabled: bool,
    grad_accumulation_steps: int,
    atom_index_batch_size: int,
    coordinatewise_loss_p: float,
):
    if not initial_dictionary.is_orthonormal and coordinatewise_loss_p > 1e-6:
        raise ValueError("For coordinatewise training, the initial dictionary must be orthonormal, consider setting coordinatewise_loss_p=0.")
    
    neural_isometry = neural_isometry.to(device)
    optim_isometry = isometry_optimizer_callable(neural_isometry.parameters())
    scheduler_isometry = isometry_scheduler_callable(optim_isometry) if isometry_scheduler_callable is not None else None
    optim_isometry.zero_grad()

    mean_function = mean_function.to(device)
    optim_mean_function = mean_function_optimizer_callable(mean_function.parameters())
    scheduler_mean_function = mean_function_scheduler_callable(optim_mean_function) if mean_function_scheduler_callable is not None else None
    optim_mean_function.zero_grad()
    
    pbar = tqdm(range(n_epochs))
    
    for epoch_i in pbar:

        if epoch_i % grad_accumulation_steps == 0:
            # TODO: add a domain sampler object
            # TODO: make this more efficient
            coords = initial_dictionary.sample_from_domain(domain_sample_size).to(device) # shape (N, d)
            energy_history_temp = []
            mean_function_mse_history_temp = []

            # use the same index samples throughout the grad accumulation steps
            # sample atom_indices as a batch of indices distributed according to a decreasing index distribution
            atom_indices = index_sampler.sample_indices(atom_index_batch_size, epoch_i, n_epochs)
            atom_indices = atom_indices.long().cpu()  # shape (A, )

            # for each function compute the inner products with all deformed atoms, thus 
            # getting a (B, A) matrix of inner products
            unique_atom_indices, atom_counts = torch.unique(atom_indices, return_counts=True)
            atom_counts = atom_counts.float().to(device)  # shape (A, )
        
        # pick batch_size (B) number of functions
        seed_batch = torch.randint(0, f_gen.n_functions if n_functions is None else n_functions, (batch_size,))
        vals = f_gen.get_batch(coords, seeds=seed_batch).to(device)  # shape (B, N)
        
        # (1: mean function) compute the mean function and do its backward:
        avg_vals = mean_function(coords).squeeze(-1)
        mean_mse = torch.mean((vals - avg_vals.unsqueeze(0)).pow(2))
        (mean_mse / grad_accumulation_steps).backward(retain_graph=False)
        mean_function_mse_history_temp.append(mean_mse.item())

        # (2: covariance training) zero-center the data and do KL expansion step:
        # get the vals centered and work with them
        vals_centered = (vals - avg_vals.unsqueeze(0)).detach()  # shape (B, N)

        coeffs_unique, deformed_functions_unique = neural_isometry.inner_products( 
            atom_indices=unique_atom_indices,
            coords=coords,
            vals=vals_centered,
            initial_dictionary=initial_dictionary,
            device=device,
            return_pullback=True,
        ) # (A, B), (A, N)
        
        # either do a coodinatewise step or a full projection step
        # TODO: can we potentially unify this? The unique one works better in some cases
        if torch.rand(1).item() < coordinatewise_loss_p:
            # project onto every coefficient to get the tensor (A, B, N)
            coordinatewise_projections = torch.einsum("ab,an->ban", coeffs_unique, deformed_functions_unique)  # shape (B, A, N)
            diff = vals_centered.unsqueeze(1) - coordinatewise_projections # shape (B, A, N)
            # per-atom MSE averaged over batch + domain: (A,)
            per_atom_mse = diff.pow(2).mean(dim=(0, 2))
            # weighted average over atoms (counts sum = atom_index_batch_size)
            energy = (per_atom_mse * atom_counts).sum() / atom_counts.sum()
        else:
            beta = initial_dictionary.gram_solve(unique_atom_indices, coeffs_unique)  # shape (A, B)
            proj = torch.matmul(beta.T, deformed_functions_unique)  # shape (B, N)
            diffs = vals_centered - proj  # shape (B, N)
            energy = torch.mean(diffs.pow(2))  # scalar

        (energy / grad_accumulation_steps).backward(retain_graph=False)
        energy_history_temp.append(energy.item())

        if (epoch_i + 1) % grad_accumulation_steps == 0:
            energy_item = sum(energy_history_temp) / len(energy_history_temp)
            mean_mse_item = sum(mean_function_mse_history_temp) / len(mean_function_mse_history_temp)
            
            if wandb_enabled:
                wandb.log({"train/energy": energy_item}, step=epoch_i)
                wandb.log({"train/mean_function_mse": mean_mse_item}, step=epoch_i)
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
                # visualize the average learning rate of the optimizer
                avg_lr1 = sum(group['lr'] for group in optim_mean_function.param_groups) / len(optim_mean_function.param_groups)
                wandb.log({"train/lr_mean_function": avg_lr1}, step=epoch_i)
                avg_lr2 = sum(group['lr'] for group in optim_isometry.param_groups) / len(optim_isometry.param_groups)
                wandb.log({"train/avg_lr_isometry": avg_lr2}, step=epoch_i)

            pbar.set_postfix({'energy': energy_item, 'mean_function_mse': mean_mse_item})
            optim_isometry.step()
            optim_mean_function.step()
            optim_isometry.zero_grad()
            optim_mean_function.zero_grad()

            if scheduler_isometry is not None:
                if isinstance(scheduler_isometry, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_isometry.step(energy_item)
                else:
                    scheduler_isometry.step()
            if scheduler_mean_function is not None:
                if isinstance(scheduler_mean_function, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler_mean_function.step(mean_mse_item)
                else:
                    scheduler_mean_function.step()
            
        
        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                wandb_enabled=wandb_enabled,
                device=device,
            )

@hydra.main(version_base=None, config_path="conf", config_name="kl_expansion")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    function_generator: FunctionClassGenerator = instantiate(conf.function_generator)  # dataset of datasets
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    index_sampler: IndexSampler = instantiate(conf.index_sampler)
    mean_function: NeuralField = instantiate(conf.mean_function)

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
        
    train(
        neural_isometry=neural_isometry,
        mean_function=mean_function,
        f_gen=function_generator,
        initial_dictionary=initial_dictionary,
        index_sampler=index_sampler,
        batch_size=conf.batch_size,
        n_epochs=conf.n_epochs,
        domain_sample_size=conf.domain_sample_size,
        device=device,
        mean_function_optimizer_callable=mean_function_optimizer_callable,
        mean_function_scheduler_callable=mean_function_scheduler_callable,
        isometry_optimizer_callable=isometry_optimizer_callable,
        isometry_scheduler_callable=isometry_scheduler_callable,
        callbacks=callbacks,
        n_functions=conf.get("n_functions", None),
        wandb_enabled=conf.wandb.enabled,
        grad_accumulation_steps=conf.grad_accumulation_steps,
        atom_index_batch_size=conf.atom_index_batch_size,
        coordinatewise_loss_p=conf.coordinatewise_loss_p,
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
