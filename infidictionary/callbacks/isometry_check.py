
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import wandb
from .base import Callback
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry

class IsometryCheck(Callback):
    """
    This callbacks checks numerical isometry of the pullback operator.
    """
    def __init__(
        self,
        dictionary: InfiDictionary,
        frequency: int,
        density: int,
        n_truncation: int | None = None,
    ):
        self.dictionary = dictionary
        self.frequency = frequency
        self.density = density
        self.n_truncation = n_truncation or self.dictionary.num_atoms
        if self.n_truncation is None:
            raise ValueError("n_truncation must be specified if the dictionary has infinite atoms.")
    
    def __call__(
        self,
        epoch: int,
        neural_isometry: NeuralIsometry,
        wandb_enabled: bool,
        device: torch.device,
    ): 
        if (epoch + 1) % self.frequency != 0 or wandb_enabled is False:
            return
        
        all_vals_original = []
        all_vals_deformed = []

        with torch.no_grad():
            n_cols = 2
            n_rows = self.n_truncation

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1:
                axes = axes.reshape(1, -1)  # make indexing consistent: axes[i, j]

            coords = self.dictionary.sample_from_domain(self.density).to(device)
            
            all_vals_original = self.dictionary.get_atom(
                coords, 
                torch.arange(self.n_truncation, device=device),
            ).to(device)

            all_vals_deformed = neural_isometry.transform(
                initial_dictionary=self.dictionary,
                atom_indices=torch.arange(self.n_truncation, device=device),
                coords=coords,
                device=device,
                mode='pullback',
            )
            
            for idx in range(self.n_truncation):
                if coords.shape[1] != 2:
                    raise ValueError("VisualizeReconstruction only supports 2D dictionaries.")

                vals_original = all_vals_original[idx] 
                vals_deformed = all_vals_deformed[idx]

                ax0, ax1 = axes[idx, 0], axes[idx, 1]
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
            inner_products = (all_vals_original @ all_vals_original.T) / all_vals_original.shape[1]
            inner_products = inner_products.detach().cpu()
            inner_products_deformed = (all_vals_deformed @ all_vals_deformed.T) / all_vals_deformed.shape[1]
            inner_products_deformed = inner_products_deformed.detach().cpu()
            fig, axes = plt.subplots(figsize=(22, 6), nrows=1, ncols=3)

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
