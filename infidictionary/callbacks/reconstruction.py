from typing import List

import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.datasets import FunctionClassGenerator
from infidictionary.neural_isometries import NeuralIsometry, IdentityIsometry
from infidictionary.utils import NeuralField

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
    atom_indices = torch.arange(prefixes[-1], device=device).long()
    avg = mean_function(coords).squeeze(-1)  # (N, )
    functions = functions - avg.unsqueeze(0)  # (B, N)
    b, all_deformed_functions = neural_isometry.inner_products( 
        atom_indices=atom_indices,
        coords=coords,
        vals=functions,
        initial_dictionary=dictionary,
        device=device,
        return_pullback=True,
    )
    all_coeffs = dictionary.gram_solve(atom_indices, b)  # shape (A, B)
    proj_list = []
    for prefix_size in prefixes:
        coeffs = all_coeffs[:prefix_size]
        deformed_functions = all_deformed_functions[:prefix_size]
        proj = coeffs.T @ deformed_functions  # shape (B, N)
        proj_list.append(proj + avg.unsqueeze(0))  # add back the mean
    return proj_list

class VisualizeReconstruction(Callback):
    """
    Visualize len(seeds) reconstruction of functions generated
    from the given FunctionClassGenerator using the provided dictionary functions
    and diffeomorphism.
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        f_gen: FunctionClassGenerator,
        seeds: list[int],
        frequency: int,
        density: int,
    ):
        self.dictionary = dictionary
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.density = density
        if self.dictionary.num_atoms is None:
            raise ValueError("VisualizeReconstruction requires a dictionary with finite number of atoms.")
    
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
            n_cols = 3
            n_rows = len(self.seeds)

            coords = self.dictionary.sample_from_domain(self.density).to(device)
            if coords.shape[1] != 2:
                raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")
            
            all_vals = []
            for i, seed in enumerate(self.seeds):
                all_vals.append(self.f_gen(coords, seed=seed).to(device))
            all_vals = torch.stack(all_vals, dim=0)  # (len(seeds), N)
            reconstructions = get_reconstructions(
                coords=coords,
                functions=all_vals,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                dictionary=self.dictionary,
                prefixes=[self.dictionary.num_atoms],
                device=device,
            )[0] # (len(seeds), N)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]
            
            for i, seed in enumerate(self.seeds):
                vals = all_vals[i]  # (N, )
                proj = reconstructions[i]  # (N, )
    
                error = torch.abs(vals - proj)
                norm2 = torch.mean(error * error).item()
                wandb.log({f"reconstruction/err_{seed}": norm2}, step=epoch)

                ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]
                cmap = "viridis"

                # ---- color scale for (Original, Projected) ----
                norm_01 = mpl.colors.Normalize(
                    vmin=vals.flatten().min(), 
                    vmax=vals.flatten().max(),
                )

                hb0 = ax0.hexbin(
                    coords[:, 0].cpu().numpy(),
                    coords[:, 1].cpu().numpy(),
                    C=vals.cpu().numpy(),
                    gridsize=50,
                    cmap=cmap,
                    norm=norm_01,
                )
                ax0.set_title("Original")
                ax0.set_ylabel(f"(seed={seed})")

                hb1 = ax1.hexbin(
                    coords[:, 0].cpu().numpy(),
                    coords[:, 1].cpu().numpy(),
                    C=proj.cpu().numpy(),
                    gridsize=50,
                    cmap=cmap,
                    norm=norm_01,
                )
                ax1.set_title("Projected")

                sm_01 = mpl.cm.ScalarMappable(norm=norm_01, cmap=cmap)
                sm_01.set_array([])
                fig.colorbar(sm_01, ax=[ax0, ax1], fraction=0.03, pad=0.02)

                # ---- color scale for (Error) ----
                row_C_2 = error.flatten()
                vmin_2 = torch.quantile(row_C_2, 0.01).item()
                vmax_2 = torch.quantile(row_C_2, 0.99).item()
                norm_2 = mpl.colors.Normalize(vmin=vmin_2, vmax=vmax_2)

                hb2 = ax2.hexbin(
                    coords[:, 0].cpu().numpy(),
                    coords[:, 1].cpu().numpy(),
                    C=error.cpu().numpy(),
                    gridsize=50,
                    cmap=cmap,
                    norm=norm_2,
                )
                ax2.set_title("Error")

                sm_2 = mpl.cm.ScalarMappable(norm=norm_2, cmap=cmap)
                sm_2.set_array([])
                fig.colorbar(sm_2, ax=[ax2], fraction=0.03, pad=0.02)

            wandb.log({f"reconstruction/visualization": wandb.Image(fig)}, step=epoch)
            plt.close(fig)



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
        if self.dictionary.num_atoms is not None:
            raise ValueError("VisualizeKLExpansionReconstruction requires a dictionary with infinite number of atoms.")
    
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

            coords = self.dictionary.sample_from_domain(self.density).to(device)
            if coords.shape[1] != 2:
                raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")
            
            all_vals = []
            for i, seed in enumerate(self.seeds):
                all_vals.append(self.f_gen(coords, seed=seed).to(device))
            all_vals = torch.stack(all_vals, dim=0)  # (len(seeds), N)

            reconstructions = get_reconstructions(
                coords=coords,
                functions=all_vals,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                dictionary=self.dictionary,
                prefixes=self.truncation_factors,
                device=device,
            ) # list of size len(self.truncation_factors) x (len(seeds), N)
            reconstructions = torch.stack(reconstructions, dim=0)  # (len(truncation_factors), len(seeds), N)
            reconstructions = reconstructions.permute(1, 0, 2)  # (len(seeds), len(truncation_factors), N)
            
            reconstructions_identity = get_reconstructions(
                coords=coords,
                functions=all_vals,
                neural_isometry=IdentityIsometry(),
                mean_function=mean_function,
                dictionary=self.dictionary,
                prefixes=self.truncation_factors,
                device=device,
            ) # list of size len(self.truncation_factors) x (len(seeds), N)
            reconstructions_identity = torch.stack(reconstructions_identity, dim=0)  # (len(truncation_factors), len(seeds), N)
            reconstructions_identity = reconstructions_identity.permute(1, 0, 2)  # (len(seeds), len(truncation_factors), N)
            
            n_cols = 1 + len(self.truncation_factors)
            n_rows = 2 * len(self.seeds)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]
            
            for i, seed in enumerate(self.seeds):
                vals = all_vals[i]  # (N, )
                projections = reconstructions[i]  # (len(truncation_factors), N)
                projections_identity = reconstructions_identity[i]  # (len(truncation_factors), N)

                error = torch.abs(vals - projections) # (len(truncation_factors), N)
                error_identity = torch.abs(vals - projections_identity) # (len(truncation_factors), N)
                error_ratio = torch.mean(error) / torch.mean(error_identity)
                wandb.log({f"reconstruction/error_ratio_{seed}": error_ratio}, step=epoch)
                wandb.log({f"reconstruction/error_{seed}": torch.mean(error).item()}, step=epoch)
                wandb.log({f"reconstruction/error_initial_dictionary_{seed}": torch.mean(error_identity).item()}, step=epoch)

                all_axes = []
                for d in [2 * i, 2 * i + 1]:
                    ax0 = axes[d, 0]
                    cmap = "viridis"
                    # ---- color scale for (Original, Projected) ----
                    norm_01 = mpl.colors.Normalize(
                        vmin=vals.flatten().min(), 
                        vmax=vals.flatten().max(),
                    )
                    hb0 = ax0.hexbin(
                        coords[:, 0].cpu().numpy(),
                        coords[:, 1].cpu().numpy(),
                        C=vals.cpu().numpy(),
                        gridsize=50,
                        cmap=cmap,
                        norm=norm_01,
                    )
                    ax0.set_title("Original")
                    ax0.set_ylabel(f"(seed={seed})")
                    all_axes.append(ax0)
                
                    for j, truncation_factor in enumerate(self.truncation_factors):
                        ax_j = axes[d, j + 1]
                        proj = projections[j] if d % 2 == 0 else projections_identity[j]
                        hb1 = ax_j.hexbin(
                            coords[:, 0].cpu().numpy(),
                            coords[:, 1].cpu().numpy(),
                            C=proj.cpu().numpy(),
                            gridsize=50,
                            cmap=cmap,
                            norm=norm_01,
                        )
                        ax_j.set_title(f"Projected with {truncation_factor} atoms ({'Identity' if d % 2 == 1 else 'Deformed'})")
                        all_axes.append(ax_j)

                sm_01 = mpl.cm.ScalarMappable(norm=norm_01, cmap=cmap)
                sm_01.set_array([])
                fig.colorbar(sm_01, ax=all_axes, fraction=0.03, pad=0.02)

            wandb.log({f"reconstruction/visualization": wandb.Image(fig)}, step=epoch)
            plt.close(fig)
