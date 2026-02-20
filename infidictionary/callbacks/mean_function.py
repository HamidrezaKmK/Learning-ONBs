
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler

class VisualizeMeanFunction(Callback):
    """
    This callbacks checks numerical isometry of the pullback operator.
    """
    # TODO: Separate the sample from domain part in this and all others
    def __init__(
        self,
        initial_dictionary: InfiDictionary,
        domain_sampler: DomainSampler,
        frequency: int,
        density: int,
    ):
        self.initial_dictionary = initial_dictionary
        self.frequency = frequency
        self.density = density
        self.domain_sampler = domain_sampler

    def __call__(
        self,
        epoch: int,
        neural_isometry: NeuralIsometry,
        mean_function: NeuralField,
        wandb_enabled: bool,
        device: torch.device,
    ): 
        if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
            return
        
        with torch.no_grad():
            coords = self.domain_sampler.sample(self.density).to(device)
            vals = mean_function(coords).squeeze(-1).to(device)
            fig, ax = plt.subplots(figsize=(5, 4))
            # do a hexbin
            hb = ax.hexbin(
                coords[:, 0].cpu(), 
                coords[:, 1].cpu(), 
                C=vals.cpu(), 
                gridsize=30, 
                cmap='viridis', 
            )
            fig.colorbar(hb, ax=ax, label='Mean Function Value')
            ax.set_title(f'Mean Function Visualization at Epoch {epoch + 1}')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            wandb.log({f'mean_function/field': wandb.Image(fig)}, step=epoch)
            plt.close()
