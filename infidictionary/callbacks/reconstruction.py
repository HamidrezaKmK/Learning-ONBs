from typing import List

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.datasets import FunctionClassGenerator
from infidictionary.neural_isometries import NeuralIsometry, IdentityIsometry
from infidictionary.utils import NeuralField
from infidictionary.domain_samplers import DomainSampler
from infidictionary.recon import get_reconstructions


class VisualizeKLExpansionReconstruction(Callback):
    """
    Visualize len(seeds) reconstruction of functions generated
    from the given FunctionClassGenerator; the difference between this and
    the original is that we get visualizations for different truncation factors
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        f_gen: FunctionClassGenerator,
        domain_sampler: DomainSampler,
        seeds: list[int],
        frequency: int,
        density: int,
        truncation_factors: list[int],
        pullback_pushforward_kwargs,
    ):
        self.dictionary = dictionary
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.density = density
        self.truncation_factors = truncation_factors
        self.domain_sampler = domain_sampler
        self.pullback_pushforward_kwargs = pullback_pushforward_kwargs
    
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
        
        with torch.no_grad():
            coords = self.domain_sampler.sample(self.density).to(device)
            
            # logic for getting the reconstructions:
            all_vals = []
            for i, seed in enumerate(self.seeds):
                all_vals.append(self.f_gen(coords, seed=seed).to(device))
            all_vals = torch.stack(all_vals, dim=0)  # (len(seeds), N, C)
            
            reconstructions = []
            reconstructions_identity = []
            for trunc in self.truncation_factors:
                recon = get_reconstructions(
                    coords=coords,
                    functions=all_vals,
                    neural_isometry=neural_isometry,
                    mean_function=mean_function,
                    dictionary=self.dictionary,
                    truncation_factor=trunc,
                    **self.pullback_pushforward_kwargs,
                ) # list of size len(self.truncation_factors) x (len(seeds), N, C)
                reconstructions.append(recon)
                recon_iden = get_reconstructions(
                    coords=coords,
                    functions=all_vals,
                    neural_isometry=IdentityIsometry(),
                    mean_function=mean_function,
                    dictionary=self.dictionary,
                    truncation_factor=trunc,
                    **self.pullback_pushforward_kwargs,
                )
                reconstructions_identity.append(recon_iden)
            reconstructions = torch.stack(reconstructions, dim=0)  # (len(truncation_factors), len(seeds), N, C)
            reconstructions = reconstructions.permute(1, 0, 2, 3)  # (len(seeds), len(truncation_factors), N, C)
            reconstructions_identity = torch.stack(reconstructions_identity, dim=0)  # (len(truncation_factors), len(seeds), N, C)
            reconstructions_identity = reconstructions_identity.permute(1, 0, 2, 3)  # (len(seeds), len(truncation_factors), N, C)

            # Setup Plotly Grid
            n_cols = 1 + len(self.truncation_factors)
            n_rows_per_seed = 2
            n_rows = n_rows_per_seed * len(self.seeds)
            
            # Create subplot titles
            subplot_titles = []
            for seed in self.seeds:
                subplot_titles.append(f"Original Signal (Seed {seed})")
                subplot_titles.extend([f"Learned Reconstructions (Trunc. #{tf})" for tf in self.truncation_factors])
                subplot_titles.append(f"Original Signal (Seed {seed})")
                subplot_titles.extend([f"Initial Reconstruction (Trunc. #{tf})" for tf in self.truncation_factors])

            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=subplot_titles,
                vertical_spacing=0.02, # Tightened for square look
                horizontal_spacing=0.02
            )

            x_np = coords[:, 0].cpu().numpy()
            y_np = coords[:, 1].cpu().numpy()

            for i, seed in enumerate(self.seeds):
                vals = all_vals[i]  # (N, )
                projections = reconstructions[i]  # (len(truncation_factors), N, C)
                projections_identity = reconstructions_identity[i]  # (len(truncation_factors), N, C)

                error = torch.abs(vals - projections) # (len(truncation_factors), N, C)
                error_identity = torch.abs(vals - projections_identity) # (len(truncation_factors), N, C)
                error = torch.mean(error, dim=-1) # (len(truncation_factors), N)
                error_identity = torch.mean(error_identity, dim=-1) # (len(truncation_factors), N)
                error_ratio = torch.mean(error) / torch.mean(error_identity)
                wandb.log({f"reconstruction/error_ratio_{seed}": error_ratio}, step=epoch)
                wandb.log({f"reconstruction/error_{seed}": torch.mean(error).item()}, step=epoch)
                wandb.log({f"reconstruction/error_initial_dictionary_{seed}": torch.mean(error_identity).item()}, step=epoch)

                # Move data to CPU once
                vals = all_vals[i].cpu().numpy()
                projs_def = reconstructions[i].cpu().numpy()
                projs_id = reconstructions_identity[i].cpu().numpy()

                # --- ROW-PAIR NORMALIZATION ---
                # Calculate min/max across all plots for this seed
                z_min = min(vals.min(), projs_def.min(), projs_id.min())
                z_max = max(vals.max(), projs_def.max(), projs_id.max())

                for row_offset in [0, 1]: 
                    curr_row = 2 * i + 1 + row_offset
                    
                    # 1. Plot Original (Column 1)
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=x_np, y=y_np, z=vals.flatten(),
                            histfunc="avg", colorscale='Viridis',
                            zmin=z_min, zmax=z_max, 
                            showscale=False, # Colorbars removed
                            nbinsx=50, nbinsy=50
                        ),
                        row=curr_row, col=1
                    )

                    # 2. Plot Truncations (Columns 2+)
                    for j, tf in enumerate(self.truncation_factors):
                        proj = projs_def[j] if row_offset == 0 else projs_id[j]
                        fig.add_trace(
                            go.Histogram2dContour(
                                x=x_np, y=y_np, z=proj.flatten(),
                                histfunc="avg", colorscale='Viridis',
                                zmin=z_min, zmax=z_max, 
                                showscale=False, # Colorbars removed
                                nbinsx=50, nbinsy=50
                            ),
                            row=curr_row, col=j + 2
                        )

                    # --- ENFORCE SQUARE ASPECT RATIO FOR THE ROW ---
                    for c in range(1, n_cols + 1):
                        fig.update_xaxes(
                            scaleanchor=f"y{((curr_row-1)*n_cols) + c}", 
                            scaleratio=1, 
                            constrain='domain',
                            row=curr_row, col=c
                        )
                        fig.update_yaxes(constrain='domain', row=curr_row, col=c)

            fig.update_layout(
                height=350 * n_rows, # Adjusted height for square containers
                width=350 * n_cols,
                title_text=f"KL Expansion Reconstruction at Epoch {epoch+1}",
                template="plotly_white",
                showlegend=False
            )

            wandb.log({f"reconstruction/0_visualization": fig}, step=epoch)
