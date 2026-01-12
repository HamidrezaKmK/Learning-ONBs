
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms.base import Diffeomorphism

class IsometryCheck(Callback):
    """
    This callbacks checks numerical isometry of the pullback operator.
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        frequency: int,
        density: int,
    ):
        self.dictionary = dictionary
        if self.dictionary.num_atoms is None:
            raise ValueError("IsometryCheck requires finite dictionary.")
        self.dictionary.compute_gram_matrix(n_domain_samples=10000, device='cpu')
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
        
        all_vals_original = []
        all_vals_deformed = []

        with torch.no_grad():
            n_cols = 2
            n_rows = self.dictionary.num_atoms

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]

            coords = self.dictionary.sample_from_domain(self.density).to(device)
            deformed_coords, logabsdet = diffeomorphism.forward(coords)

            for i, idx in enumerate(range(self.dictionary.num_atoms)):
                if coords.shape[1] != 2:
                    raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")

                vals_original = self.dictionary.get_atom(coords, idx).to(device)
                all_vals_original.append(vals_original.cpu())

                vals_deformed = self.dictionary.get_atom(deformed_coords, idx).to(device)
                vals_deformed = vals_deformed * torch.exp(0.5 * logabsdet)
                all_vals_deformed.append(vals_deformed.cpu())

                ax0, ax1 = axes[i, 0], axes[i, 1]
                cmap = "viridis"

                vmin_01 = torch.quantile(vals_original.flatten(), 0.01).item()
                vmax_01 = torch.quantile(vals_original.flatten(), 0.99).item()
                norm_01 = mpl.colors.Normalize(vmin=vmin_01, vmax=vmax_01)

                hb0 = ax0.hexbin(
                    coords[:, 0].cpu().numpy(),
                    coords[:, 1].cpu().numpy(),
                    C=vals_original.cpu().numpy(),
                    gridsize=50,
                    cmap=cmap,
                    norm=norm_01,
                )
                ax0.set_title("Original")
                ax0.set_ylabel(f"(idx={idx})")

                sm_01 = mpl.cm.ScalarMappable(norm=norm_01, cmap=cmap)
                sm_01.set_array([])
                fig.colorbar(sm_01, ax=ax0, fraction=0.03, pad=0.02)

                vmin_02 = torch.quantile(vals_deformed.flatten(), 0.01).item()
                vmax_02 = torch.quantile(vals_deformed.flatten(), 0.99).item()
                norm_02 = mpl.colors.Normalize(vmin=vmin_02, vmax=vmax_02)

                hb1 = ax1.hexbin(
                    coords[:, 0].cpu().numpy(),
                    coords[:, 1].cpu().numpy(),
                    C=vals_deformed.cpu().numpy(),
                    gridsize=50,
                    cmap=cmap,
                    norm=norm_02,
                )
                ax1.set_title("Deformed")

                sm_02 = mpl.cm.ScalarMappable(norm=norm_02, cmap=cmap)
                sm_02.set_array([])
                fig.colorbar(sm_02, ax=ax1, fraction=0.03, pad=0.02)

            wandb.log({f"isometry_check/before_and_after": wandb.Image(fig)})
            plt.close(fig)

            # Log overall statistics in one plot as a heatmap
            # compute pairwise inner products
            inner_products = torch.zeros((len(all_vals_original), len(all_vals_original)))
            inner_products_deformed = torch.zeros((len(all_vals_original), len(all_vals_original)))
            fig, axes = plt.subplots(figsize=(22, 6), nrows=1, ncols=3)
            for i in range(len(all_vals_original)):
                for j in range(len(all_vals_original)):
                    vi = all_vals_original[i]
                    vj = all_vals_original[j]
                    inner_products[i, j] = (vi * vj).mean().item()
                    vi_def = all_vals_deformed[i]
                    vj_def = all_vals_deformed[j]
                    inner_products_deformed[i, j] = (vi_def * vj_def).mean().item()
            # heatmap of inner products
            m = axes[2].imshow(inner_products_deformed)
            fig.colorbar(m, ax=axes[2], label='Inner Product (Deformed)')
            axes[2].set_title('Pairwise Inner Products (Deformed)')
            axes[2].set_xlabel('Function Index')
            axes[2].set_ylabel('Function Index')

            m = axes[1].imshow(inner_products)
            fig.colorbar(m, ax=axes[1], label='Inner Product')
            axes[1].set_title('Pairwise Inner Products (Original)')
            axes[1].set_xlabel('Function Index')
            axes[1].set_ylabel('Function Index')

            diffs = inner_products - inner_products_deformed
            axes[0].hist(diffs.flatten(), bins=30)
            axes[0].set_title('Differences in Inner Products (Original - Deformed)')
            axes[0].set_xlabel('Difference')
            axes[0].set_ylabel('Frequency')

            wandb.log({f"isometry_check/inner_products": wandb.Image(fig)})
            plt.close(fig)
