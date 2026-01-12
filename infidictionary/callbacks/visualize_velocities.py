
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms import CTDiffeomorphism

class VisualizeVelocityField(Callback):
    """
    Visualize the velocity field of a continuous-time normalizing flow
    at 9 different timesteps.
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        timesteps: list[float],
        frequency: int,
        density: int,
        viz_magnitude: float = 1.0,
    ):
        self.timesteps = timesteps
        self.frequency = frequency
        self.density = density
        self.dictionary = dictionary
        self.viz_magnitude = viz_magnitude
        if len(timesteps) != 9:
            raise ValueError("VisualizeVelocityField requires exactly 9 timesteps.")

    
    def __call__(
        self,
        epoch: int,
        diffeomorphism: CTDiffeomorphism,
        wandb_enabled: bool,
        device: torch.device,
    ): 
        if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
            return
        
        N = self.density
        with torch.no_grad():
            fig, axes = plt.subplots(3, 3, figsize=(5 * 3, 4 * 3))
            xy = self.dictionary.sample_from_domain(N).to(device)

            t = torch.tensor(self.timesteps, device=device)
            t, _ = torch.sort(t)
            t = torch.repeat_interleave(t, N)
            xy = xy.repeat(len(self.timesteps), 1)
            v =  self.viz_magnitude * diffeomorphism.velocity_field(t, xy)

            for row in range(3):
                for col in range(3):
                    ax = axes[row, col]
                    idx = row * 3 + col
                    ax.quiver(xy[idx * N:(idx + 1) * N,0].detach().cpu(), xy[idx * N:(idx + 1) * N,1].detach().cpu(),
                            v[idx * N:(idx + 1) * N,0].detach().cpu(), v[idx * N:(idx + 1) * N,1].detach().cpu(),
                            angles='xy', scale_units='xy', scale=10.0, alpha=1)
                    ax.set_title(f'Velocity Field at t={t[idx * N].item():.2f}')
                    ax.set_xlim(-1.05,1.05)
                    ax.set_ylim(-1.05,1.05)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
            wandb.log({f"velocity_field/visualization": wandb.Image(fig)})
            plt.close(fig)
