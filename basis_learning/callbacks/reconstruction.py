
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from basis_learning.bases.base import BaseFunction
from basis_learning.datasets import FunctionClassGenerator
from basis_learning.diffeomorphisms.base import Diffeomorphism

class VisualizeReconstruction(Callback):
    """
    Visualize len(seeds) reconstruction of functions generated
    from the given FunctionClassGenerator using the provided basis functions
    and diffeomorphism.
    """
    def __init__(
        self,
        basis: BaseFunction,
        indices: list[int],
        f_gen: FunctionClassGenerator,
        seeds: list[int],
        frequency: int,
        density: int,
    ):
        self.basis = basis
        self.indices = indices
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.density = density

    
    def __call__(
        self,
        epoch: int,
        diffeomorphism: Diffeomorphism,
        loss: float, 
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
                coords = self.basis.sample_from_domain(self.density).to(device)
                if coords.shape[1] != 2:
                    raise ValueError("VisualizeReconstruction only supports 2D bases.")
                vals = self.f_gen(coords, seed=seed).to(device)

                proj = torch.zeros_like(vals, device=device)
                for idx in self.indices:
                    deformed_coords, logabsdet = diffeomorphism.forward(coords)
                    deformed_vals = self.basis.get(deformed_coords, idx).to(device)
                    deformed_vals = deformed_vals * torch.exp(0.5 * logabsdet)

                    inner_product = torch.mean(deformed_vals * vals)
                    proj += inner_product * deformed_vals

                error = torch.abs(vals - proj)
                norm2 = torch.mean(error * error).item()
                wandb.log({f"reconstruction/err_{seed}": norm2})

                ax0, ax1, ax2 = axes[i, 0], axes[i, 1], axes[i, 2]
                cmap = "viridis"

                # ---- color scale for (Original, Projected) ----
                row_C_01 = torch.cat([vals.flatten(), proj.flatten()])
                vmin_01 = torch.quantile(row_C_01, 0.01).item()
                vmax_01 = torch.quantile(row_C_01, 0.99).item()
                norm_01 = mpl.colors.Normalize(vmin=vmin_01, vmax=vmax_01)

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
