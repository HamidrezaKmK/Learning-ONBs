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

class FFNeuralField(NeuralField):
    def __init__(self, input_dim, output_dim, n_features=64, sigma=10.0,
                 hidden_dims=(256,256), activation=nn.ReLU):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.ff = FourierFeatures(input_dim, n_features, sigma=sigma)
        layers = []
        prev = 2 * n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        z = self.ff(coords)
        return self.net(z)


class TimeEmbedding(nn.Module, ABC):
    @property
    @abstractmethod
    def out_dim(self) -> int:
        raise NotImplementedError("Subclasses must implement this method")
    
class SinusoidalTimeEmbedding(TimeEmbedding):
    """
    Scalar t  ->  [sin(w_k t), cos(w_k t)]_k  (Fourier features)
    Similar in spirit to what diffusion models use.
    """
    def __init__(self, num_frequencies: int = 8, max_log_freq: float = 3.0):
        super().__init__()
        self.num_frequencies = num_frequencies

        # Frequencies: 2^0, 2^{max_log_freq} on a log scale
        freqs = torch.exp(torch.linspace(0.0, max_log_freq, num_frequencies) * math.log(2.0))
        self.register_buffer("freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 * self.num_frequencies

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)              # (B, 1)

        # (B, 1, num_freqs)
        angles = t[..., None] * self.freqs[None, None, :] * 2 * math.pi
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 1, 2*num_freqs)

        return emb.view(t.size(0), -1)       # (B, 2 * num_freqs)
    