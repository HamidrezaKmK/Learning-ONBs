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
from infidictionary.utils import pairwise_inner_product

def get_reconstructions(
    coords: torch.Tensor, # (N, d)
    functions: torch.Tensor, # (B, N)
    neural_isometry: NeuralIsometry,
    mean_function: NeuralField,
    dictionary: InfiDictionary,
    prefixes: List[int],
    device: torch.device,
):
    """
    Get the reconstructions of the given functions using the provided
    neural isometry and dictionary at the specified atom indices (prefixes).
    """
    all_sub_indices = []
    all_atom_indices = dictionary.get_truncated_indices(prefixes[-1]).to(device)
    for prefix in prefixes:
        atom_indices = dictionary.get_truncated_indices(prefix).to(device)
        # find the indices in all_atom_indices that correspond to atom_indices
        mask = torch.zeros(all_atom_indices.shape[0], dtype=torch.bool, device=device)
        for idx in atom_indices:
            match = (all_atom_indices == idx).all(dim=1)  # find the row that matches idx
            mask = mask | match  # update the mask to include this index
        sub_indices = torch.where(mask)[0]  # get the indices of all_atom_indices that correspond to atom_indices
        all_sub_indices.append(sub_indices)
    
    avg = mean_function(coords)  # (N, C)
    functions = functions - avg.unsqueeze(0)  # (B, N, C)
    src_coords, src_logabsdet, src_pulled_back = neural_isometry.pullback(
        tgt_coords=coords,
        tgt_logabsdet=torch.zeros(coords.shape[0], device=device),
        tgt_field=functions,
        start_time=0,
        end_time=1,
    )
    dictionary_values = dictionary.get_atoms(
        src_coords,
        all_atom_indices,
    )  # shape (A, N, C)
    _, _, dictionary_pushforward = neural_isometry.pushforward(
        src_coords=src_coords,
        src_logabsdet=src_logabsdet,
        src_field=dictionary_values,
        start_time=0,
        end_time=1,
    )
    coefficients = pairwise_inner_product(
        src_pulled_back,
        dictionary_values,
        src_logabsdet,
    ) # shape (B, A)
    reconstructions = []
    for i in range(len(prefixes)):
        c = coefficients[:, all_sub_indices[i]] # shape (B, A_prefix)
        dict_values = dictionary_pushforward[all_sub_indices[i], :, :] # shape (A_prefix, N, C)
        recon = c @ dict_values.view(dict_values.shape[0], -1) # shape (B, N * C)
        recon = recon.view(functions.shape) # shape (B, N, C)
        recon = recon + avg.unsqueeze(0) # add back the mean
        reconstructions.append(recon)
    return reconstructions

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
    ):
        self.dictionary = dictionary
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.density = density
        self.truncation_factors = truncation_factors
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
        
        with torch.no_grad():
            coords = self.domain_sampler.sample(self.density).to(device)
            
            # logic for getting the reconstructions:
            all_vals = []
            for i, seed in enumerate(self.seeds):
                all_vals.append(self.f_gen(coords, seed=seed).to(device))
            all_vals = torch.stack(all_vals, dim=0)  # (len(seeds), N, C)

            reconstructions = get_reconstructions(
                coords=coords,
                functions=all_vals,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                dictionary=self.dictionary,
                prefixes=self.truncation_factors,
                device=device,
            ) # list of size len(self.truncation_factors) x (len(seeds), N, C)
            reconstructions = torch.stack(reconstructions, dim=0)  # (len(truncation_factors), len(seeds), N, C)
            reconstructions = reconstructions.permute(1, 0, 2, 3)  # (len(seeds), len(truncation_factors), N, C)
            
            reconstructions_identity = get_reconstructions(
                coords=coords,
                functions=all_vals,
                neural_isometry=IdentityIsometry(),
                mean_function=mean_function,
                dictionary=self.dictionary,
                prefixes=self.truncation_factors,
                device=device,
            ) # list of size len(self.truncation_factors) x (len(seeds), N, C)
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

    # def __call__(
    #     self,
    #     epoch: int,
    #     neural_isometry: NeuralIsometry,
    #     mean_function: NeuralField,
    #     wandb_enabled: bool,
    #     device: torch.device,
    # ): 
    #     if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
    #         return
        
    #     with torch.no_grad():

    #         coords = self.domain_sampler.sample(self.density).to(device)
    #         if coords.shape[1] != 2:
    #             raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")
            
    #         all_vals = []
    #         for i, seed in enumerate(self.seeds):
    #             all_vals.append(self.f_gen(coords, seed=seed).to(device))
    #         all_vals = torch.stack(all_vals, dim=0)  # (len(seeds), N, C)

    #         reconstructions = get_reconstructions(
    #             coords=coords,
    #             functions=all_vals,
    #             neural_isometry=neural_isometry,
    #             mean_function=mean_function,
    #             dictionary=self.dictionary,
    #             prefixes=self.truncation_factors,
    #             device=device,
    #         ) # list of size len(self.truncation_factors) x (len(seeds), N, C)
    #         reconstructions = torch.stack(reconstructions, dim=0)  # (len(truncation_factors), len(seeds), N, C)
    #         reconstructions = reconstructions.permute(1, 0, 2, 3)  # (len(seeds), len(truncation_factors), N, C)
            
    #         reconstructions_identity = get_reconstructions(
    #             coords=coords,
    #             functions=all_vals,
    #             neural_isometry=IdentityIsometry(),
    #             mean_function=mean_function,
    #             dictionary=self.dictionary,
    #             prefixes=self.truncation_factors,
    #             device=device,
    #         ) # list of size len(self.truncation_factors) x (len(seeds), N, C)
    #         reconstructions_identity = torch.stack(reconstructions_identity, dim=0)  # (len(truncation_factors), len(seeds), N, C)
    #         reconstructions_identity = reconstructions_identity.permute(1, 0, 2, 3)  # (len(seeds), len(truncation_factors), N, C)
            
    #         n_cols = 1 + len(self.truncation_factors)
    #         n_rows = 2 * len(self.seeds)
    #         fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    #         if n_rows == 1:
    #             axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]
            
    #         for i, seed in enumerate(self.seeds):
    #             vals = all_vals[i]  # (N, )
    #             projections = reconstructions[i]  # (len(truncation_factors), N, C)
    #             projections_identity = reconstructions_identity[i]  # (len(truncation_factors), N, C)

    #             error = torch.abs(vals - projections) # (len(truncation_factors), N, C)
    #             error_identity = torch.abs(vals - projections_identity) # (len(truncation_factors), N, C)
    #             error = torch.mean(error, dim=-1) # (len(truncation_factors), N)
    #             error_identity = torch.mean(error_identity, dim=-1) # (len(truncation_factors), N)
    #             error_ratio = torch.mean(error) / torch.mean(error_identity)
    #             wandb.log({f"reconstruction/error_ratio_{seed}": error_ratio}, step=epoch)
    #             wandb.log({f"reconstruction/error_{seed}": torch.mean(error).item()}, step=epoch)
    #             wandb.log({f"reconstruction/error_initial_dictionary_{seed}": torch.mean(error_identity).item()}, step=epoch)

    #             all_axes = []
    #             for d in [2 * i, 2 * i + 1]:
    #                 ax0 = axes[d, 0]
    #                 cmap = "viridis"
    #                 # ---- color scale for (Original, Projected) ----
    #                 norm_01 = mpl.colors.Normalize(
    #                     vmin=vals.flatten().min(), 
    #                     vmax=vals.flatten().max(),
    #                 )
    #                 hb0 = ax0.hexbin(
    #                     coords[:, 0].cpu().numpy(),
    #                     coords[:, 1].cpu().numpy(),
    #                     C=vals.cpu().numpy(),
    #                     gridsize=50,
    #                     cmap=cmap,
    #                     norm=norm_01,
    #                 )
    #                 ax0.set_title("Original")
    #                 ax0.set_ylabel(f"(seed={seed})")
    #                 all_axes.append(ax0)
                
    #                 for j, truncation_factor in enumerate(self.truncation_factors):
    #                     ax_j = axes[d, j + 1]
    #                     proj = projections[j] if d % 2 == 0 else projections_identity[j]
    #                     hb1 = ax_j.hexbin(
    #                         coords[:, 0].cpu().numpy(),
    #                         coords[:, 1].cpu().numpy(),
    #                         C=proj.cpu().numpy(),
    #                         gridsize=50,
    #                         cmap=cmap,
    #                         norm=norm_01,
    #                     )
    #                     ax_j.set_title(f"Projected with {truncation_factor} atoms ({'Identity' if d % 2 == 1 else 'Deformed'})")
    #                     all_axes.append(ax_j)

    #             sm_01 = mpl.cm.ScalarMappable(norm=norm_01, cmap=cmap)
    #             sm_01.set_array([])
    #             fig.colorbar(sm_01, ax=all_axes, fraction=0.03, pad=0.02)

    #         wandb.log({f"reconstruction/visualization": wandb.Image(fig)}, step=epoch)
    #         plt.close(fig)
