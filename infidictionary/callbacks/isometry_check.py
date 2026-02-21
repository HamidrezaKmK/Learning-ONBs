
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler
from infidictionary.utils import pairwise_inner_product

class IsometryCheck(Callback):
    """
    This callbacks checks numerical isometry of the pullback operator.
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        domain_sampler: DomainSampler,
        frequency: int,
        density: int,
        indices: list,
    ):
        self.dictionary = dictionary
        self.frequency = frequency
        self.density = density
        self.domain_sampler = domain_sampler
        self.indices = torch.tensor(indices, dtype=torch.long)
    
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
            # pushforward logic:
            src_field = self.dictionary.get_atoms(
                coords, 
                self.indices.to(device),
            ).to(device) # shape (A, N, C)

            src_logabsdet = torch.zeros(coords.shape[0], device=device) # shape (N, )
            tgt_coords, tgt_logabsdet, tgt_field = neural_isometry.pushforward(
                src_coords=coords,
                src_logabsdet=src_logabsdet,
                src_field=src_field,
                start_time=0,
                end_time=1,
            )

            # --- PART 1: Spatial Visualizations (Before vs. After) ---
            n_rows_spatial = len(self.indices)
            fig_spatial = make_subplots(
                rows=n_rows_spatial, cols=2,
                subplot_titles=("Original", "Deformed") * n_rows_spatial,
                horizontal_spacing=0.1
            )

            x_np, y_np = coords[:, 0].cpu().numpy(), coords[:, 1].cpu().numpy()

            for idx in range(len(self.indices)):
                v_orig = src_field[idx].cpu().numpy().flatten()
                v_def = tgt_field[idx].cpu().numpy().flatten()

                # Robust scaling using quantiles (as in your original code)
                vmin_orig = torch.quantile(src_field[idx], 0.01).item()
                vmax_orig = torch.quantile(src_field[idx], 0.99).item()
                vmin_def = torch.quantile(tgt_field[idx], 0.01).item()
                vmax_def = torch.quantile(tgt_field[idx], 0.99).item()

                # Original Column
                fig_spatial.add_trace(
                    go.Histogram2dContour(
                        x=x_np, y=y_np, z=v_orig, histfunc="avg",
                        colorscale='Viridis', zmin=vmin_orig, zmax=vmax_orig,
                        nbinsx=50, nbinsy=50, showscale=False,
                    ), row=idx+1, col=1
                )

                # Deformed Column
                fig_spatial.add_trace(
                    go.Histogram2dContour(
                        x=x_np, y=y_np, z=v_def, histfunc="avg",
                        colorscale='Viridis', zmin=vmin_def, zmax=vmax_def,
                        nbinsx=50, nbinsy=50, showscale=False,
                    ), row=idx+1, col=2
                )

                fig_spatial.update_xaxes(scaleanchor=f"y{idx*2 + 1}", scaleratio=1, row=idx+1, col=1)
                fig_spatial.update_yaxes(scaleanchor=f"x{idx*2 + 1}", scaleratio=1, row=idx+1, col=1)
                
                fig_spatial.update_xaxes(scaleanchor=f"y{idx*2 + 2}", scaleratio=1, row=idx+1, col=2)
                fig_spatial.update_yaxes(scaleanchor=f"x{idx*2 + 2}", scaleratio=1, row=idx+1, col=2)

            fig_spatial.update_layout(height=400 * n_rows_spatial, width=1000, template="plotly_white")
            wandb.log({"isometry_check/before_and_after": fig_spatial}, step=epoch)

            # --- PART 2: Isometry Heatmaps ---
            inner_orig = pairwise_inner_product(src_field, src_field, src_logabsdet).cpu().numpy()
            inner_def = pairwise_inner_product(tgt_field, tgt_field, tgt_logabsdet).cpu().numpy()
            diffs = (inner_orig - inner_def).flatten()

            fig_iso = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Inner Product Diff Dist.", "Original Inner Products", "Deformed Inner Products"),
                # Adjust spacing so they don't overlap when square
                horizontal_spacing=0.1 
            )

            # 1. Histogram of differences
            fig_iso.add_trace(go.Histogram(x=diffs, name="Diffs", nbinsx=30), row=1, col=1)

            # 2. Heatmaps
            for i, data in enumerate([inner_orig, inner_def]):
                col_idx = i + 2
                fig_iso.add_trace(
                    go.Heatmap(
                        z=data, 
                        colorscale='RdBu', 
                        zmid=0,
                        showscale=True if i == 1 else False # Only show one colorbar to save space
                    ), row=1, col=col_idx
                )
                
                # FORCE SQUARE: Anchor the y-axis to the x-axis for this specific subplot
                # In a 1x3 grid, col 2 is yaxis2/xaxis2, col 3 is yaxis3/xaxis3
                fig_iso.update_xaxes(scaleanchor=f"y{col_idx}", scaleratio=1, row=1, col=col_idx)

            # Adjust layout: Keep width/height ratio reasonable for squares
            fig_iso.update_layout(
                height=500, 
                width=1200, 
                title_text="Isometry Verification Heatmaps",
                template="plotly_white"
            )

            wandb.log({"isometry_check/inner_products": fig_iso}, step=epoch)
