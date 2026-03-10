import torch
import wandb
from plotly.subplots import make_subplots
from .base import Callback
from .plot_utils import make_trace
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.neural_isometries import NeuralIsometry, IdentityIsometry
from infidictionary.utils import NeuralField
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
        f_gen,
        seeds: list[int],
        frequency: int,
        truncation_factors: list[int],
        pullback_pushforward_kwargs,
        identity_isometry: NeuralIsometry,
    ):
        self.dictionary = dictionary
        self.f_gen = f_gen
        self.seeds = seeds
        self.frequency = frequency
        self.truncation_factors = truncation_factors
        self.pullback_pushforward_kwargs = pullback_pushforward_kwargs
        self.identity_isometry = identity_isometry
    
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
        
        neural_isometry.eval()
        mean_function.eval()
        with torch.no_grad():
            # Fetch per-seed coordinates and values from the function generator
            seed_data = []
            for seed in self.seeds:
                coords, vals = self.f_gen(seed)
                seed_data.append((coords.to(device), vals.to(device)))

            # Per-seed reconstructions: list of (T, N_i, C) tensors
            all_reconstructions = []
            all_reconstructions_identity = []
            for coords, vals in seed_data:
                functions = vals.unsqueeze(0)  # (1, N, C)
                recons = []
                recons_iden = []
                for trunc in self.truncation_factors:
                    recon = get_reconstructions(
                        coords=coords,
                        functions=functions,
                        neural_isometry=neural_isometry,
                        mean_function=mean_function,
                        dictionary=self.dictionary,
                        truncation_factor=trunc,
                        **self.pullback_pushforward_kwargs,
                    ).squeeze(0)  # (N, C)
                    recons.append(recon)
                    recon_iden = get_reconstructions(
                        coords=coords,
                        functions=functions,
                        neural_isometry=self.identity_isometry,
                        mean_function=mean_function,
                        dictionary=self.dictionary,
                        truncation_factor=trunc,
                        **self.pullback_pushforward_kwargs,
                    ).squeeze(0)  # (N, C)
                    recons_iden.append(recon_iden)
                all_reconstructions.append(torch.stack(recons, dim=0))       # (T, N, C)
                all_reconstructions_identity.append(torch.stack(recons_iden, dim=0))  # (T, N, C)

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

            for i, seed in enumerate(self.seeds):
                coords, vals = seed_data[i]
                projections = all_reconstructions[i]          # (T, N, C)
                projections_identity = all_reconstructions_identity[i]  # (T, N, C)

                x_np = coords[:, 0].cpu().numpy()
                y_np = coords[:, 1].cpu().numpy()

                error = torch.abs(vals - projections) # (T, N, C)
                error_identity = torch.abs(vals - projections_identity) # (T, N, C)
                error = torch.mean(error, dim=-1) # (T, N)
                error_identity = torch.mean(error_identity, dim=-1) # (T, N)
                error_ratio = torch.mean(error) / torch.mean(error_identity)
                wandb.log({f"reconstruction/error_ratio_{seed}": error_ratio}, step=epoch)
                wandb.log({f"reconstruction/error_{seed}": torch.mean(error).item()}, step=epoch)
                wandb.log({f"reconstruction/error_initial_dictionary_{seed}": torch.mean(error_identity).item()}, step=epoch)

                # Move data to CPU once
                vals = vals.cpu().numpy()
                projs_def = projections.cpu().numpy()
                projs_id = projections_identity.cpu().numpy()

                # --- ROW-PAIR NORMALIZATION ---
                # Calculate min/max across all plots for this seed
                z_min = min(vals.min(), projs_def.min(), projs_id.min())
                z_max = max(vals.max(), projs_def.max(), projs_id.max())

                for row_offset in [0, 1]:
                    curr_row = 2 * i + 1 + row_offset

                    # 1. Plot Original (Column 1)
                    fig.add_trace(
                        make_trace(x_np, y_np, vals, z_min, z_max),
                        row=curr_row, col=1
                    )

                    # 2. Plot Truncations (Columns 2+)
                    for j, tf in enumerate(self.truncation_factors):
                        proj = projs_def[j] if row_offset == 0 else projs_id[j]
                        fig.add_trace(
                            make_trace(x_np, y_np, proj, z_min, z_max),
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

        neural_isometry.train()
        mean_function.train()
