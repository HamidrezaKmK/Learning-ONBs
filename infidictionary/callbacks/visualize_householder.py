
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Callable
import wandb
from .base import Callback
from infidictionary.neural_isometries import NeuralIsometry, HouseholderIsometry
from infidictionary.utils import NeuralField

class VisualizeHouseholderReflections(Callback):
    """
    Visualize the householder reflection fields of the householder isometries.
    """
    def __init__(
        self,
        sample_from_domain: Callable,
        reflection_indices: list[list[int]],
        frequency: int,
        density: int,
    ):
        self.frequency = frequency
        self.density = density
        self.sample_from_domain = sample_from_domain 
        self.reflection_indices = reflection_indices

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
        # iterate over the submodules of neural_isometry to find a HalfDensityIsometry
        module_index = 1
        for module in neural_isometry.modules():
            if isinstance(module, HouseholderIsometry):
                hh_isometry = module
                
                N = self.density
                xy = self.sample_from_domain(N).to(device)
                with torch.no_grad():
                    nrows = len(self.reflection_indices) 
                    ncols = len(self.reflection_indices[0])

                    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * nrows, 4 * ncols))
                    
                    for row in range(nrows):
                        for col in range(ncols):
                            ax = axes[row, col]
                            real_idx = self.reflection_indices[row][col]
                            ax.set_title(f'Reflection Field {real_idx}')
                            ax.set_xlim(-1.05,1.05)
                            ax.set_ylim(-1.05,1.05)
                            ax.set_xlabel('x')
                            ax.set_ylabel('y')

                            # add a colorbar
                            hh_nef = hh_isometry._modules[f"reflection_{real_idx}"]
                            vals = hh_nef(xy).squeeze(-1).to(device)
                            hb = ax.hexbin(
                                xy[:, 0].cpu(), 
                                xy[:, 1].cpu(), 
                                C=vals.cpu(), 
                                gridsize=30, 
                                cmap='viridis', 
                            )
                            fig.colorbar(hb, ax=ax, label='values')

                    wandb.log({f"reflection_field/visualization_{module_index}": wandb.Image(fig)}, step=epoch)
                    plt.close(fig)
