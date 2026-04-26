import torch
from torch import nn

from .base import TimeEvolvingField, NerfFourierFeatures, _build_mlp


class NerfSpatioTemporalField(TimeEvolvingField):
    """
    Spatial residual field returning U: (N, R, C).
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        rank: int,
        time_emb_dim: int,
        spatial_hidden_dims: tuple = (256, 256, 256),
        activation=nn.SiLU,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.C = output_dim
        self.R = rank

        self.nerf_features = NerfFourierFeatures(
            coords_dim, nerf_n_levels, nerf_n_base, nerf_freq_min, nerf_freq_max,
        )
        ff_spatial_dim = coords_dim + self.nerf_features.out_dim

        self.ff = _build_mlp(
            time_emb_dim + ff_spatial_dim,
            rank * output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=False,
        )

    def forward(self, t_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Returns U: (N, R, C). t_emb: (N, emb_dim)."""
        N = x.shape[0]
        spatial = torch.cat([t_emb, x, self.nerf_features(x)], dim=-1)
        return self.ff(spatial).view(N, self.R, self.C)
