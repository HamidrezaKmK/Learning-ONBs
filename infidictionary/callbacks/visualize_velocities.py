
import torch
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import wandb
from .base import Callback

from infidictionary.domain_samplers import DomainSampler
from infidictionary.neural_isometries import NeuralIsometry, LagrangianIsometry
from infidictionary.networks import NeuralField

class VisualizeVelocityField(Callback):
    """
    Visualize the velocity field of a continuous-time normalizing flow
    at 9 different timesteps.
    """
    def __init__(
        self,
        domain_sampler: DomainSampler,
        timesteps: list[float],
        frequency: int,
        density: int,
        viz_magnitude: float = 1.0,
    ):
        self.timesteps = timesteps
        self.frequency = frequency
        self.density = density
        self.domain_sampler = domain_sampler
        self.viz_magnitude = viz_magnitude
        if len(timesteps) != 9:
            raise ValueError("VisualizeVelocityField requires exactly 9 timesteps.")

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
            if isinstance(module, LagrangianIsometry) and hasattr(module, "diffeomorphism") and hasattr(module.diffeomorphism, "velocity_field"):
                hd_isometry = module

                with torch.no_grad():
                    xy = self.domain_sampler.sample(self.density).to(device)
                    N = xy.shape[0]

                    t = torch.tensor(self.timesteps, device=device)
                    t, _ = torch.sort(t)
                    t = torch.repeat_interleave(t, N)
                    xy = xy.repeat(len(self.timesteps), 1)
                    v = hd_isometry.diffeomorphism.velocity_field(t, xy)

                    sorted_timesteps = sorted(self.timesteps)
                    fig = make_subplots(
                        rows=3, cols=3,
                        subplot_titles=[f"t={ts:.2f}" for ts in sorted_timesteps],
                        vertical_spacing=0.05,
                        horizontal_spacing=0.05,
                    )

                    for idx in range(9):
                        row = idx // 3 + 1
                        col = idx % 3 + 1

                        x_np = xy[idx * N:(idx + 1) * N, 0].cpu().numpy()
                        y_np = xy[idx * N:(idx + 1) * N, 1].cpu().numpy()
                        u_np = v[idx * N:(idx + 1) * N, 0].cpu().numpy()
                        v_np = v[idx * N:(idx + 1) * N, 1].cpu().numpy()

                        quiver = ff.create_quiver(x_np, y_np, u_np, v_np, scale=0.1, arrow_scale=self.viz_magnitude)
                        for trace in quiver.data:
                            fig.add_trace(trace, row=row, col=col)

                        axis_idx = (row - 1) * 3 + col
                        fig.update_xaxes(scaleanchor=f"y{axis_idx}", scaleratio=1, range=[-1.05, 1.05], constrain='domain', row=row, col=col)
                        fig.update_yaxes(range=[-1.05, 1.05], constrain='domain', row=row, col=col)

                    fig.update_layout(
                        height=400 * 3,
                        width=400 * 3,
                        title_text=f"Velocity Field at Epoch {epoch + 1}",
                        template="plotly_white",
                        showlegend=False,
                    )

                    wandb.log({f"velocity_field/visualization_{module_index}": fig}, step=epoch)
