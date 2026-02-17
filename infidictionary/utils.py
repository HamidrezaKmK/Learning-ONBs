import torch
from torch import nn
from typing import Callable, Dict, Any
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

class TimeEvolvingField(NeuralField):
    def __init__(
        self,
        base_field_partial: Callable[[Dict[str, Any]], NeuralField],
        coords_dim: int = 2,
        output_dim: int = 1,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.time_embedding = SinusoidalTimeEmbedding() # TODO: fix this!
        self.time_evolving_field = base_field_partial(
            input_dim=self.time_embedding.out_dim + coords_dim,
            output_dim=output_dim,
        )
        self.coords_dim = coords_dim
        self.output_dim = output_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (B, )
        # x: (B, d)
        t_emb = self.time_embedding(t.unsqueeze(-1))  # (B, time_embedding_dim)
        inp = torch.cat([t_emb, x], dim=-1)  # (B, time_embedding_dim + d)
        return self.time_evolving_field(inp)  # (B, output_dim)
 
def pairwise_inner_product(
    f1: torch.Tensor, # (A, N, C) or (N, C)
    f2: torch.Tensor, # (B, N, C) or (N, C)
    logabsdet: torch.Tensor | None = None, # (N, )
) -> torch.Tensor:
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    if f1.dim() == 2:
        f1 = f1.unsqueeze(0)  # (1, N, C)
    if f2.dim() == 2:
        f2 = f2.unsqueeze(0)  # (1, N, C)
    ret = torch.einsum("anc, bnc->ab", f1 * torch.exp(logabsdet)[None, :, None], f2) # (A, B)
    ret = ret.squeeze() / f1.shape[1] # (A, B) or (A,) or (B,) or scalar
    # if scaler make it (1, )
    if ret.dim() == 0:
        ret = ret.unsqueeze(0)
    return ret

def norm2(
    f: torch.Tensor, # (B, N, C) or (N, C)
    logabsdet: torch.Tensor | None = None, # (N, )
) -> torch.Tensor:
    if f.dim() == 2:
        f = f.unsqueeze(0)  # (1, N, C)
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f.shape[1], device=f.device, dtype=f.dtype)
    ret = torch.einsum("bnc, bnc->b", f * torch.exp(logabsdet)[None, :, None], f) # (B, )
    return ret.squeeze() / f.shape[1]  # (B,) or scalar

def parallel_inner_product(
    f1: torch.Tensor, # (B, N, C)
    f2: torch.Tensor, # (B, N, C)
    logabsdet: torch.Tensor | None = None, # (B, N, ) or (N, ) or None
) -> torch.Tensor:
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    if logabsdet.dim() == 1:
        logabsdet = logabsdet.unsqueeze(0).repeat(f1.shape[0], 1)  # (B, N)
    ret = torch.einsum("bnc, bnc->b", f1 * torch.exp(logabsdet)[:, :, None], f2) # (B, )
    return ret.squeeze() / f1.shape[1]  # (B,) or scalar
