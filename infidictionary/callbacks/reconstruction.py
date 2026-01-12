
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.datasets import FunctionClassGenerator
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.utils import gram_projection

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
        if self.dictionary.num_atoms is None:
            raise ValueError("VisualizeReconstruction requires finite dictionary.")
        self.dictionary.compute_gram_matrix(n_domain_samples=10000, device='cpu')
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.density = density

    
    def __call__(
        self,
        epoch: int,
        diffeomorphism: Diffeomorphism,
        wandb_enabled: bool,
        device: torch.device,
    ): 
        if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
            return
        
        with torch.no_grad():
            n_cols = 3
            n_rows = len(self.seeds)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]

            for i, seed in enumerate(self.seeds):
                coords = self.dictionary.sample_from_domain(self.density).to(device)
                if coords.shape[1] != 2:
                    raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")
                vals = self.f_gen(coords, seed=seed).to(device)
                warped_coords, logabsdets = diffeomorphism.forward(coords)
                proj = gram_projection(
                    coords=coords,
                    warped_coords=warped_coords,
                    logabsdets=logabsdets,
                    vals=vals,
                    initial_dictionary=self.dictionary,
                    device=device,
                )

                error = torch.abs(vals - proj)
                norm2 = torch.mean(error * error).item()
                wandb.log({f"reconstruction/err_{seed}": norm2})

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

            wandb.log({f"reconstruction/visualization": wandb.Image(fig)})
            plt.close(fig)
