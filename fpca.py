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
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler

# Add resolver for hydra
OmegaConf.register_new_resolver("eval", eval)
# torch.set_default_dtype(torch.float64)

def get_grad_norm(
    model: torch.nn.Module,
):
    # visualize norm of the gradients of the parameters
    all_grads = []
    for param in model.parameters():
        if param.grad is not None:
            all_grads.append(param.grad.view(-1))
    all_grads = torch.cat(all_grads)
    grad_norm = torch.norm(all_grads).item()
    return grad_norm

def get_param_norm(
    model: torch.nn.Module,
):
    all_params = []
    for param in model.parameters():
        all_params.append(param.view(-1))
    all_params = torch.cat(all_params)
    param_norm = torch.norm(all_params).item()
    return param_norm

def train(
    neural_isometry: NeuralIsometry,
    mean_function: NeuralField,
    f_gen: FunctionClassGenerator,
    initial_dictionary: InfiDictionary,
    domain_sampler: DomainSampler,
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
    energy_estimation_kwargs: dict,
    model_state_kwargs: dict,
    pullback_pushforward_kwargs: dict,
):  
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
            neural_isometry.shuffle_model_state(**model_state_kwargs)
            coords = domain_sampler.sample(domain_sample_size).to(device) # shape (N, d)
            energy_history_temp = []
            mean_function_mse_history_temp = []
        
        # pick batch_size (B) number of functions
        seed_batch = torch.randint(0, f_gen.n_functions if n_functions is None else n_functions, (batch_size,))
        vals = f_gen.get_batch(coords, seeds=seed_batch).to(device)  # shape (B, N, C)
        
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

        if (epoch_i + 1) % grad_accumulation_steps == 0:
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

@hydra.main(version_base=None, config_path="conf", config_name="fpca")
def main(conf: DictConfig):

    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    function_generator: FunctionClassGenerator = instantiate(conf.function_generator)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    mean_function: NeuralField = instantiate(conf.mean_function)
    domain_sampler: DomainSampler = instantiate(conf.domain_sampler)

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
        domain_sampler=domain_sampler,
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
        energy_estimation_kwargs=conf.get("energy_estimation_kwargs", {}),
        model_state_kwargs=conf.get("model_state_kwargs", {}),
        pullback_pushforward_kwargs=conf.get("pullback_pushforward_kwargs", {}),
    )

    if conf.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    main()
