import torch
from torch import nn
from typing import Dict, Any
from abc import ABC, abstractmethod
import math


class NeuralField(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, coords):
        raise NotImplementedError("Subclasses must implement this method")

class TimeEvolvingField(NeuralField):
    @abstractmethod
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement this method")
    
class MLPNeuralField(NeuralField):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Dict[Any, int], activation=nn.ReLU):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims.values():
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, coords):
        return self.network(coords)


class FourierFeatures(nn.Module):
    def __init__(self, input_dim=2, n_features=64, sigma=10.0):
        super().__init__()
        # B ~ N(0, sigma^2)
        B = torch.randn(input_dim, n_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        # x: (N, 2)
        proj = 2 * math.pi * x @ self.B  # (N, n_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2*n_features)


class RMSNorm(nn.Module):
    """Per-sample RMS normalisation over the feature dimension.

    Normalises each row x to x / RMS(x), where RMS(x) = sqrt(mean(x²)).
    No learnable affine parameters — the downstream linear layer provides
    all necessary rescaling.  Unlike BatchNorm this is sample-independent:
    identical behaviour in train and eval, works at any batch size (N=1).
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return x / (rms + self.eps)


def _init_orthogonal(module: nn.Module) -> None:
    """Orthogonal weight init for Linear layers; zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class NerfFourierFeatures(nn.Module):
    """Multi-scale random Fourier features with a NeRF-inspired frequency allocation.

    Frequency levels are logarithmically spaced from ``freq_min`` to ``freq_max``.
    At each level ``l`` the number of random projections is::

        n_l = max(1, n_base // 2^l)

    so the lowest-frequency level contributes ``n_base`` projections (= 2*n_base
    sin/cos features) and each subsequent level halves the count.  This biases the
    feature vector toward low-frequency content while still covering high frequencies,
    matching the NeRF intuition that coarse structure matters more than fine detail.

    Total output dimension: ``2 * sum_l n_l`` (sin + cos per projection, all levels).

    Args:
        input_dim:  Coordinate dimension d.
        n_levels:   Number of frequency levels L.
        n_base:     Projections at the lowest-frequency level (halved each level).
        freq_min:   Frequency sigma at level 0.
        freq_max:   Frequency sigma at level L-1 (levels are log-spaced).
    """

    def __init__(
        self,
        input_dim: int,
        n_levels: int = 8,
        n_base: int = 64,
        freq_min: float = 1.0,
        freq_max: float = 64.0,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.input_dim = input_dim

        # Log-spaced sigmas from freq_min to freq_max.
        log_freqs = torch.linspace(math.log(freq_min), math.log(freq_max), n_levels)
        sigmas = log_freqs.exp().tolist()

        # Projections per level: n_base, n_base//2, n_base//4, ..., ≥1.
        self._n_per_level = [max(1, n_base >> l) for l in range(n_levels)]

        for l, (sigma, n) in enumerate(zip(sigmas, self._n_per_level)):
            B = torch.randn(input_dim, n) * sigma
            self.register_buffer(f"B_{l}", B)

        self.out_dim = 2 * sum(self._n_per_level)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for l in range(self.n_levels):
            B = getattr(self, f"B_{l}")
            proj = 2 * math.pi * x @ B          # (N, n_l)
            feats.append(torch.sin(proj))
            feats.append(torch.cos(proj))
        return torch.cat(feats, dim=-1)          # (N, out_dim)



def _build_mlp(
    in_dim: int, 
    out_dim: int, 
    hidden_dims: tuple, 
    activation, 
    use_batchnorm: bool, 
    use_rmsnorm: bool, 
    bias: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h, bias=bias))
        if use_rmsnorm:
            layers.append(RMSNorm())
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim, bias=bias))
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_dim, affine=False))
    return nn.Sequential(*layers)
