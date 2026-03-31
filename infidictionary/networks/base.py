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
