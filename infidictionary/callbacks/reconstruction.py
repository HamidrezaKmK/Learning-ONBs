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
        if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
            return
        
        with torch.no_grad():

            coords = self.domain_sampler.sample(self.density).to(device)
            if coords.shape[1] != 2:
                raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")
            
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
            
            n_cols = 1 + len(self.truncation_factors)
            n_rows = 2 * len(self.seeds)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]
            
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
