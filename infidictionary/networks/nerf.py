import torch
from torch import nn

from .base import ConditionalField, NerfFourierFeatures, NeuralField, _build_mlp


class _NerfBackbone(nn.Module):
    """Shared coordinate encoder and MLP for conditional and unconditional fields."""

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        cond_dim: int = 0,
        hidden_dims: tuple = (256, 256, 256),
        activation=nn.SiLU,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
    ):
        super().__init__()
        self.cond_dim = int(cond_dim)
        self.nerf_features = NerfFourierFeatures(
            coords_dim, nerf_n_levels, nerf_n_base, nerf_freq_min, nerf_freq_max,
        )
        encoded_dim = self.cond_dim + coords_dim + self.nerf_features.out_dim
        self.network = _build_mlp(
            encoded_dim,
            output_dim,
            hidden_dims,
            activation,
            use_rmsnorm=True,
            bias=False,
        )

    def forward(self, coords: torch.Tensor, cond_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.cond_dim == 0:
            if cond_emb is not None:
                raise ValueError("An unconditional NeRF backbone does not accept cond_emb")
            inputs = [coords, self.nerf_features(coords)]
        else:
            if cond_emb is None:
                raise ValueError("A conditional NeRF backbone requires cond_emb")
            if cond_emb.shape != (coords.shape[0], self.cond_dim):
                raise ValueError(
                    "cond_emb must have shape "
                    f"({coords.shape[0]}, {self.cond_dim}); got {tuple(cond_emb.shape)}"
                )
            inputs = [cond_emb, coords, self.nerf_features(coords)]
        return self.network(torch.cat(inputs, dim=-1))


class NerfNeuralField(NeuralField):
    """Unconditional NeRF field for mean-function estimation."""

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        activation=nn.SiLU,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.backbone = _NerfBackbone(
            coords_dim=coords_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            nerf_n_levels=nerf_n_levels,
            nerf_n_base=nerf_n_base,
            nerf_freq_min=nerf_freq_min,
            nerf_freq_max=nerf_freq_max,
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.backbone(coords)


class NerfConditionalField(ConditionalField):
    """Conditional NeRF field returning tensors with shape ``(N, R, C)``."""

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        rank: int,
        cond_dim: int,
        hidden_dims: tuple = (256, 256, 256),
        activation=nn.SiLU,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.C = output_dim
        self.R = rank
        self.backbone = _NerfBackbone(
            coords_dim=coords_dim,
            output_dim=rank * output_dim,
            cond_dim=cond_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            nerf_n_levels=nerf_n_levels,
            nerf_n_base=nerf_n_base,
            nerf_freq_min=nerf_freq_min,
            nerf_freq_max=nerf_freq_max,
        )

    def forward(self, cond_emb: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        return self.backbone(coords, cond_emb).view(coords.shape[0], self.R, self.C)
