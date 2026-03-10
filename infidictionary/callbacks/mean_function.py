
import torch
import wandb
import plotly.graph_objects as go
from .base import Callback
from .plot_utils import make_trace
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler

class VisualizeMeanFunction(Callback):
    """
    This callbacks checks numerical isometry of the pullback operator.
    """
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
        if (epoch + 1) % self.frequency != 0 or not wandb_enabled:
            return
        mean_function.eval()
        with torch.no_grad():
            coords = self.domain_sampler.sample(self.density).to(device)
            vals = mean_function(coords).to(device)

            x = coords[:, 0].cpu().numpy()
            y = coords[:, 1].cpu().numpy()
            data = vals.cpu().numpy()  # (N, C)

            fig = go.Figure(make_trace(x, y, data))

            fig.update_xaxes(
                scaleanchor="y",
                scaleratio=1,
                constrain='domain'
            )
            fig.update_yaxes(
                constrain='domain'
            )

            fig.update_layout(
                title=f'Mean Function Visualization at Epoch {epoch + 1}',
                xaxis_title='x',
                yaxis_title='y',
                width=600,
                height=600, # Set width and height equal for a square container
                template="plotly_white"
            )

            wandb.log({f'mean_function/field': fig}, step=epoch)
        mean_function.train()
