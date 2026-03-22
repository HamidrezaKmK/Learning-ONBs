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


class NTKMLPNeuralField(NeuralField):
    """MLP with NTK parameterization for lazy-training / kernel stability.

    Weights W_l ~ N(0, 1); each pre-activation is divided by sqrt(fan_in).
    This keeps every hidden layer output O(1), making each gradient step change
    f by O(1/sqrt(m)), so the NTK remains approximately constant during training.

    Use smooth activations (default: Tanh) — ReLU causes discrete Jacobian jumps
    when activation patterns flip, which destabilises the kernel.

    Plain MLPs on raw coordinates suffer from spectral bias: the NTK has
    exponentially small eigenvalues for high frequencies, so the network can only
    learn smooth low-frequency functions.  Setting ``n_fourier_features > 0``
    prepends a frozen random Fourier embedding γ(x) = [sin(Bx), cos(Bx)] before
    the MLP.  The NTK of the combined network K(x,x') ≈ K_MLP(γ(x), γ(x')) has
    large eigenvalues across all frequencies covered by B, enabling high-frequency
    reconstruction while the frozen features keep the MLP weights in the NTK regime.

    Args:
        input_dim:         coordinate dimension.
        output_dim:        output channels.
        hidden_dims:       dict or list of hidden widths.
        activation:        activation constructor (default nn.Tanh).
        n_fourier_features: number of random Fourier feature pairs (0 = disabled).
        fourier_sigma:     bandwidth of the random Fourier features.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims,
        activation=nn.Tanh,
        n_fourier_features: int = 128,
        fourier_sigma: float = 10.0,
    ):
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        # Explicitly cast to Python int: OmegaConf IntegerNode values are not
        # accepted by PyTorch's nn.Linear when Hydra instantiates the class.
        # Use hasattr(.values) to handle both plain dict and OmegaConf DictConfig
        # (isinstance(x, dict) returns False for DictConfig).
        if hasattr(hidden_dims, 'values'):
            widths = [int(v) for v in hidden_dims.values()]
        else:
            widths = [int(v) for v in hidden_dims]

        if n_fourier_features > 0:
            self.ff = FourierFeatures(input_dim, n_fourier_features, sigma=fourier_sigma)
            mlp_input_dim = 2 * n_fourier_features
        else:
            self.ff = None
            mlp_input_dim = input_dim

        dims = [int(mlp_input_dim)] + widths + [int(output_dim)]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.activation = activation()
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
            nn.init.zeros_(layer.bias)

    def forward(self, coords):
        x = self.ff(coords) if self.ff is not None else coords
        for layer in self.layers[:-1]:
            # Divide the pre-activation by sqrt(fan_in), not sqrt(current width).
            # This keeps every hidden layer output O(1) regardless of the previous
            # layer's width — critical for the first layer where fan_in = input_dim
            # (e.g. 2 or 2*n_fourier_features), not the hidden width (e.g. 1024).
            x = self.activation(layer(x) / math.sqrt(layer.in_features))
        return self.layers[-1](x) / math.sqrt(self.layers[-1].in_features)


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

def _init_orthogonal(module: nn.Module) -> None:
    """Orthogonal weight init for Linear layers; zero bias."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FactoredTimeEvolvingField(NeuralField):
    """v(t, x) = post_mlp( BN(spatial_mlp(FF(x))) @ W(t) ).

    Factored design: spatial and temporal pathways are completely separate.
      - spatial_mlp:  FF(x) → g(x) ∈ ℝᴰ.  BatchNorm1d(affine=False) keeps Σ_g ≈ I
                      (safe: spatial features are t-independent).
      - time_mlp:     sinusoidal_emb(t) → vec(W) ∈ ℝᴰˣᶜ, reshaped to (N, D, C).
                      Divided by per-sample Frobenius norm so ‖W(t_i)‖_F = 1.
    At convergence: E_x[‖g @ W‖²] ≈ ‖W‖²_F = 1 for every t.
    An optional post-combination MLP applies a residual nonlinearity on top;
    its last layer is zero-initialised so it starts as identity.
    All Linear layers use orthogonal weight initialisation.

    Args:
        coords_dim:            coordinate dimension d.
        output_dim:            output channels C.
        spatial_hidden_dims:   hidden widths for the spatial MLP.
        time_hidden_dims:      hidden widths for the time MLP.
        feature_dim:           output width D of the spatial MLP (= rows of W).
        use_fourier_features:  use FourierFeatures(x) as spatial input.
        n_fourier_features:    Fourier feature pairs (spatial input = 2×this).
        fourier_sigma:         bandwidth of random Fourier features.
        n_time_freqs:          sinusoidal frequency count for the time embedding.
        activation:            activation constructor (default nn.ReLU).
        post_combination_dims: hidden widths for post-combination MLP; empty = no MLP.
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
        activation=nn.ReLU,
        post_combination_dims: tuple = (),
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        D, C = feature_dim, output_dim
        self._D = D
        self._C = C
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        # ── Spatial MLP + BN ─────────────────────────────────────────────────
        if use_fourier_features:
            self.ff = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
            spatial_in = 2 * n_fourier_features + coords_dim  # FF(x) ++ x
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
        self.spatial_mlp.apply(_init_orthogonal)
        self.spatial_bn = nn.BatchNorm1d(D, affine=False)

        # ── Time MLP → W(t) ∈ ℝᴺˣᴰˣᶜ ────────────────────────────────────────
        time_in = self.time_embedding.out_dim
        time_layers = []
        prev = time_in
        for h in time_hidden_dims:
            time_layers += [nn.Linear(prev, h), activation()]
            prev = h
        time_layers += [nn.Linear(prev, D * C)]
        self.time_mlp = nn.Sequential(*time_layers)
        self.time_mlp.apply(_init_orthogonal)

        # ── Post-combination nonlinear MLP (optional, with residual) ─────────
        # Last linear is zero-initialised so the residual starts as identity.
        if post_combination_dims:
            layers = []
            prev = C
            for h in post_combination_dims:
                lin = nn.Linear(prev, h)
                nn.init.orthogonal_(lin.weight)
                nn.init.zeros_(lin.bias)
                layers += [lin, activation()]
                prev = h
            last = nn.Linear(prev, C)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            layers += [last]
            self.post_combination = nn.Sequential(*layers)
        else:
            self.post_combination = None

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,), x: (N, d)
        x_in = torch.cat([self.ff(x), x], dim=-1) if self.ff is not None else x  # (N, spatial_in)
        g = self.spatial_bn(self.spatial_mlp(x_in))                        # (N, D)

        t_emb = self.time_embedding(t)                                      # (N, D_t)
        W_raw = self.time_mlp(t_emb).reshape(-1, self._D, self._C)         # (N, D, C)
        W = W_raw / (W_raw.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)

        v = torch.einsum('nd,ndc->nc', g, W)                               # (N, C)

        if self.post_combination is not None:
            v = v + self.post_combination(v)

        return v


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
