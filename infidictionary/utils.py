import torch
from torch import nn
from typing import Callable, Dict, Any
from abc import ABC, abstractmethod
import math
import functools

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


class HybridNeuralField(NeuralField):
    """Hybrid neural field: sum of an MLP branch (raw coords) and an FF branch (Fourier features)."""
    def __init__(self, input_dim, output_dim, n_features=64, sigma=10.0,
                 ff_hidden_dims=(256, 256), mlp_hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        mlp_hidden_dict = {i: d for i, d in enumerate(mlp_hidden_dims)}
        self.mlp_branch = MLPNeuralField(input_dim, output_dim, hidden_dims=mlp_hidden_dict, activation=activation)
        self.ff_branch = FFNeuralField(input_dim, output_dim, n_features=n_features, sigma=sigma,
                                       hidden_dims=ff_hidden_dims, activation=activation)

    def forward(self, coords):
        return self.mlp_branch(coords) + self.ff_branch(coords)


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

class FactoredTimeEvolvingField(NeuralField):
    """v(t, x) = spatial_bn(spatial_mlp(x)) @ W(t),
    where W(t) = reshape(time_mlp(t)) / norm_denominator.

    The two networks are completely separate:
      - spatial_mlp: x (or FourierFeatures(x)) → g(x) ∈ ℝᴰ, wide MLP.
      - time_mlp:    sinusoidal_emb(t) → vec(W) ∈ ℝᴰˣᶜ.

    BatchNorm1d(affine=False) on the spatial output drives Σ_g → I via running
    buffers (independent of t because spatial_mlp takes only x).

    Two normalisation modes for W, selected by `norm_mode`:
      "instantaneous"  — divide by the current ‖W_raw‖_F.  Exact unit Frobenius
                         norm at every step, but gradient can be large when the
                         norm is transiently small.
      "running_ema"    — divide by a running EMA of ‖W_raw‖_F, updated without
                         gradients (like BN's running variance).  The denominator
                         is smooth and bounded away from zero, giving stable
                         gradients at the cost of approximate normalisation.

    At convergence: E_x[‖v(t,x)‖²] = tr(Wᵀ Σ_g W) ≈ ‖W‖²_F ≈ 1 for every t.

    Args:
        coords_dim:           coordinate dimension d.
        output_dim:           output channels C.
        spatial_hidden_dims:  widths of hidden layers in the spatial MLP.
        time_hidden_dims:     widths of hidden layers in the time MLP.
        feature_dim:          output width D of the spatial MLP.
        use_fourier_features: pass FourierFeatures(x) into the spatial MLP instead of x.
        n_fourier_features:   number of Fourier feature pairs (spatial input = 2×this).
        fourier_sigma:        bandwidth of random Fourier features.
        n_time_freqs:         sinusoidal frequency count for the time embedding.
        norm_mode:            "instantaneous" or "running_ema".
        norm_momentum:        EMA momentum (only used when norm_mode="running_ema").
        activation:           activation constructor (default nn.ReLU).
    """
    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        spatial_hidden_dims: tuple = (256, 256, 256, 256),
        time_hidden_dims: tuple = (256, 256),
        feature_dim: int = 256,
        use_fourier_features: bool = True,
        n_fourier_features: int = 64,
        fourier_sigma: float = 10.0,
        n_time_freqs: int = 8,
        norm_mode: str = "running_ema",
        norm_momentum: float = 0.1,
        activation=nn.ReLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        assert norm_mode in ("instantaneous", "running_ema"), \
            f"norm_mode must be 'instantaneous' or 'running_ema', got '{norm_mode}'"
        D, C = feature_dim, output_dim
        self.norm_mode = norm_mode
        self.norm_momentum = norm_momentum
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        # ── Spatial MLP ───────────────────────────────────────────────────────
        if use_fourier_features:
            self.ff = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
            spatial_in = 2 * n_fourier_features
        else:
            self.ff = None
            spatial_in = coords_dim

        spatial_layers = []
        prev = spatial_in
        for h in spatial_hidden_dims:
            spatial_layers += [nn.Linear(prev, h), activation()]
            prev = h
        spatial_layers += [nn.Linear(prev, D)]
        self.spatial_mlp = nn.Sequential(*spatial_layers)

        # affine=False: no learnable γ/β — BN running buffers track E_x[g_c]
        # and Var_x(g_c) purely over the spatial distribution, independent of t.
        self.spatial_bn = nn.BatchNorm1d(D, affine=False)

        # ── Time MLP ──────────────────────────────────────────────────────────
        time_in = self.time_embedding.out_dim
        time_layers = []
        prev = time_in
        for h in time_hidden_dims:
            time_layers += [nn.Linear(prev, h), activation()]
            prev = h
        time_layers += [nn.Linear(prev, D * C)]
        self.time_mlp = nn.Sequential(*time_layers)

        self._D = D
        self._C = C

        # Running EMA of ‖W_raw‖_F (used only when norm_mode="running_ema").
        # Initialised to 1 so the network starts with approximately unit-norm W.
        self.register_buffer('running_W_norm', torch.ones(1))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,) — all identical at each Householder step
        # x: (N, d)

        # Spatial features; BN running buffers keep Σ_g ≈ I
        x_in = self.ff(x) if self.ff is not None else x
        g = self.spatial_bn(self.spatial_mlp(x_in))             # (N, D)

        # Time weight matrix; all N share the same t, so one W per forward call
        t_emb = self.time_embedding(t[0:1])                      # (1, D_t)
        W_raw = self.time_mlp(t_emb).reshape(self._D, self._C)  # (D, C)

        if self.norm_mode == "instantaneous":
            # Exact unit Frobenius norm; gradient can spike when norm is small
            W = W_raw / (W_raw.norm() + 1e-8)
        else:
            # Update running EMA without gradients (stable denominator)
            if self.training:
                with torch.no_grad():
                    self.running_W_norm.mul_(1 - self.norm_momentum).add_(
                        W_raw.norm() * self.norm_momentum
                    )
            W = W_raw / (self.running_W_norm + 1e-8)

        return g @ W                                             # (N, C)


class TimeEvolvingField(NeuralField):
    def __init__(
        self,
        base_field_partial: Callable[[Dict[str, Any]], NeuralField],
        coords_dim: int = 2,
        output_dim: int = 1,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.time_embedding = SinusoidalTimeEmbedding()
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
    first_unsqueezed = False
    second_unsqueezed = False
    if f1.dim() == 2:
        f1 = f1.unsqueeze(0)  # (1, N, C)
        first_unsqueezed = True
    if f2.dim() == 2:
        f2 = f2.unsqueeze(0)  # (1, N, C)
        second_unsqueezed = True
    ret = torch.einsum("anc, bnc->ab", f1 * torch.exp(logabsdet)[None, :, None], f2) # (A, B)
    ret = ret / f1.shape[1] # (A, B) or (A,) or (B,) or scalar
    if first_unsqueezed:
        ret = ret.squeeze(0)
    if second_unsqueezed:
        ret = ret.squeeze(-1)
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
