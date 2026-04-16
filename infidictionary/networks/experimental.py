import itertools
import math
import torch
from torch import nn

from .base import TimeEvolvingField, _init_orthogonal, RMSNorm, FourierFeatures
from .time_embedding import SinusoidalTimeEmbedding
from infidictionary.dictionaries import FourierDictionary


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


class FourierDistortedFieldV4(TimeEvolvingField):
    """Frequency-sweeping Fourier field with a fixed random direction bank.

    Compared to ``FourierDistortedFieldV2``, all time-dependent MLPs that
    produced directions, phases and feature masks are replaced by fixed random
    buffers sampled once at construction time.  The only time-dependent
    computation is the scalar frequency schedule ``_compute_freq``, which sweeps
    the spatial frequency from ``low_freq`` to ``high_freq`` as t increases.

    At each point (t_n, x_n) the two output streams are::

        # fixed buffers
        W_orth   ∈ R^{n_modes × K × d}   (unit-norm directions)
        phases   ∈ R^{n_modes × K}        (uniform random in [0, 1))

        # per-sample, time-dependent scalars
        freq1(t) ∈ R^{K/2}               (coarse → fine)
        freq2(t) ∈ R^{K/2}               (fine → coarse, complementary)

        proj_i = W_orth[:, i, :] · x     # (N, n_modes)
        arg_i  = 2π · freq_i(t) · (proj_i + phases_i)
        feat_i = [sin(arg_i), cos(arg_i)].flatten()   # (N, n_modes * K)

        field_i = ff_i( cat([t_emb, x, feat_i]) )

    The two streams use the ``K/2`` directions and phases in the first and
    second halves of the K axis respectively, so they always probe orthogonal
    subsets of the direction bank.

    Args:
        coords_dim:          Spatial dimension d.
        output_dim:          Output channels C.
        n_modes:             Number of random directions (total bank size n_modes×K).
        freq_strides:        Frequency stride steps; K = 2*freq_strides.
        spatial_hidden_dims: Hidden widths for the two spatial MLPs (ff1, ff2).
        n_time_freqs:        Sinusoidal time-embedding frequency count.
        low_freq / high_freq: Frequency sweep range.
        low_t / high_t:      Time range corresponding to the sweep.
        gamma:               Ramp shape (>0 slow-start, <0 fast-start).
        activation:          Activation constructor.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_modes: int,
        freq_strides: int = 1,
        spatial_hidden_dims: tuple = (256, 256, 256),
        n_time_freqs: int = 256,
        low_freq: float = 0.0,
        high_freq: float = 10.0,
        low_t: float = 0.0,
        high_t: float = 1.0,
        gamma: float = 1.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.n_modes   = n_modes
        self.C         = output_dim
        self.d         = coords_dim
        self.low_freq  = low_freq
        self.high_freq = high_freq
        self.low_t     = low_t
        self.high_t    = high_t
        self.gamma     = gamma
        self.freq_strides = freq_strides
        self.K = 2 * freq_strides  # K/2 directions per stream

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        # Fixed random unit directions: (n_modes, K, d).
        W = torch.randn(n_modes, self.K, coords_dim)
        W = W / W.norm(dim=-1, keepdim=True)
        self.register_buffer('W_orth', W)

        # Fixed random phases uniform in [0, 1): (n_modes, K).
        self.register_buffer('phases', torch.rand(n_modes, self.K))

        ff_in_dim = emb_dim + coords_dim + n_modes * self.K
        self.ff1 = self._build_mlp(
            ff_in_dim, output_dim, spatial_hidden_dims, activation,
            use_batchnorm=True, use_rmsnorm=True, bias=True,
        )
        self.ff2 = self._build_mlp(
            ff_in_dim, output_dim, spatial_hidden_dims, activation,
            use_batchnorm=True, use_rmsnorm=True, bias=True,
        )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dims: tuple, activation,
                   use_batchnorm: bool, use_rmsnorm: bool, bias: bool) -> nn.Sequential:
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

    def _compute_freq(self, t: torch.Tensor):
        t_scaled = ((t.clamp(self.low_t, self.high_t) - self.low_t)
                    / (self.high_t - self.low_t + 1e-3)).unsqueeze(-1)
        t_scaled = (t_scaled
                    + torch.arange(self.freq_strides, device=t.device).unsqueeze(0)
                    / self.freq_strides)
        t_scaled = t_scaled % 1.0
        t_scaled = torch.cat([t_scaled, 1 - t_scaled], dim=-1)  # (N, K)
        if self.gamma < 0:
            freq_scale = 1 - (1 - t_scaled) ** (-self.gamma)
        else:
            freq_scale = t_scaled ** self.gamma
        freq = self.low_freq + freq_scale * (self.high_freq - self.low_freq)
        return freq[:, :self.K // 2], freq[:, self.K // 2:]

    def forward(self, t: torch.Tensor, x: torch.Tensor):
        N = x.shape[0]
        t_emb = self.time_embedding(t)                           # (N, emb_dim)

        # Project x onto all fixed directions: (N, n_modes, K)
        proj = torch.einsum('mkd,nd->nmk', self.W_orth, x)

        freq1, freq2 = self._compute_freq(t)                     # (N, K/2) each

        # Stream 1: first K/2 slots, coarse→fine frequency schedule
        K_half = self.K // 2
        proj1   = proj[:, :, :K_half]                            # (N, n_modes, K/2)
        phases1 = self.phases[:, :K_half]                        # (n_modes, K/2)
        arg1    = 2 * math.pi * freq1.unsqueeze(1) * (proj1 + phases1)
        feat1   = torch.cat([torch.sin(arg1), torch.cos(arg1)], dim=-1)  # (N, n_modes, K)
        feat1   = feat1.reshape(N, self.n_modes * self.K)

        # Stream 2: second K/2 slots, fine→coarse (complementary) schedule
        proj2   = proj[:, :, K_half:]                            # (N, n_modes, K/2)
        phases2 = self.phases[:, K_half:]                        # (n_modes, K/2)
        arg2    = 2 * math.pi * freq2.unsqueeze(1) * (proj2 + phases2)
        feat2   = torch.cat([torch.sin(arg2), torch.cos(arg2)], dim=-1)  # (N, n_modes, K)
        feat2   = feat2.reshape(N, self.n_modes * self.K)

        field1 = self.ff1(torch.cat([t_emb, x, feat1], dim=-1))
        field2 = self.ff2(torch.cat([t_emb, x, feat2], dim=-1))

        return field1, field2

class FourierDistortedFieldV5(TimeEvolvingField):
    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_truncation: int,
        top_k: int,
        spatial_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        n_fourier_features: int = 64,
        fourier_sigma: float = 10.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.C = output_dim
        self.d = coords_dim
        self.top_k = top_k
        
        self.fourier_dict = FourierDictionary(
            domain_dim=coords_dim,
            num_channels=output_dim,
            p=0.3,
        )
        indices_of_interest = self.fourier_dict.get_truncated_indices(n_truncation)
        n_atoms = len(indices_of_interest)
        # Store indices for the alpha atoms
        self.register_buffer("indices_alpha", indices_of_interest)
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim
        
        # Atom embeddings used as "keys" for the attention mechanism to select alpha atoms
        self.atom_embeddings = nn.Parameter(torch.randn(n_atoms, emb_dim) / math.sqrt(emb_dim))
        
        # Fixed random Fourier features for beta's generic network
        self.fourier_features = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
        ff_spatial_dim = coords_dim + 2 * n_fourier_features  # cat(x, FF(x))
        
        # Beta MLP
        self.ff_beta = self._build_mlp(
            emb_dim + ff_spatial_dim,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dims: tuple, activation, use_batchnorm: bool, use_rmsnorm: bool, bias: bool) -> nn.Sequential:
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        # Time query: (N, emb_dim)
        t_emb = self.time_embedding(t)
        
        # --- Alpha: Attention-weighted sum of initial atoms ---
        with torch.no_grad():
            # Evaluate the base atoms at the given coordinates x
            # Shape: (n_atoms, N, C)
            alpha_vals = self.fourier_dict.get_atoms(x.detach(), self.indices_alpha).detach()
        
        # Compute attention scores
        scores = torch.matmul(t_emb, self.atom_embeddings.T) / math.sqrt(self.atom_embeddings.shape[-1])
        
        # --- Top-K Sparsity Masking ---
        # Find the top-k scores and their indices
        topk_vals, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        # Create a tensor of -inf to mask out the non-top-k elements
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(dim=-1, index=topk_indices, src=topk_vals)
        
        # Softmax will now output exactly 0 for everything outside the top-k
        attn_weights = torch.softmax(sparse_scores, dim=-1)
        
        # Combine the atoms using the sparse attention weights
        alpha = torch.einsum('nk,knc->nc', attn_weights, alpha_vals)  # (N, C)
        
        # --- Beta: Generic Fourier Feature Network ---
        # Get high-frequency spatial content for beta
        ff_feats = self.fourier_features(x)                             # (N, 2*n_ff)
        spatial_in = torch.cat([x, ff_feats], dim=-1)                   # (N, d + 2*n_ff)
        
        # Feed time embedding and spatial features into the beta MLP
        beta = self.ff_beta(torch.cat([t_emb, spatial_in], dim=-1))     # (N, C)
        
        return alpha, beta
    


class FourierDistortedFieldV6(TimeEvolvingField):
    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_truncation: int,
        top_k: int,
        spatial_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        nerf_n_levels: int = 8,
        nerf_n_base: int = 64,
        nerf_freq_min: float = 1.0,
        nerf_freq_max: float = 64.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.C = output_dim
        self.d = coords_dim
        self.top_k = top_k

        self.fourier_dict = FourierDictionary(
            domain_dim=coords_dim,
            num_channels=output_dim,
            p=0.3,
        )
        indices_of_interest = self.fourier_dict.get_truncated_indices(n_truncation)
        n_atoms = len(indices_of_interest)
        # Store indices for the alpha atoms
        self.register_buffer("indices_alpha", indices_of_interest)

        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        # NeRF-style multi-scale Fourier features for beta: more projections at
        # low frequencies, halving each level up to nerf_n_levels.
        self.nerf_features = NerfFourierFeatures(
            coords_dim, nerf_n_levels, nerf_n_base, nerf_freq_min, nerf_freq_max,
        )
        ff_spatial_dim = coords_dim + self.nerf_features.out_dim  # cat(x, nerf(x))

        # Alpha MLP
        self.feature_selector = self._build_mlp(
            emb_dim,
            self.C * n_atoms,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

        # Beta MLP
        self.ff_beta = self._build_mlp(
            emb_dim + ff_spatial_dim,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dims: tuple, activation, use_batchnorm: bool, use_rmsnorm: bool, bias: bool) -> nn.Sequential:
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        # Time query: (N, emb_dim)
        t_emb = self.time_embedding(t)
        
        # --- Alpha: Attention-weighted sum of initial atoms ---
        with torch.no_grad():
            # Evaluate the base atoms at the given coordinates x
            # Shape: (n_atoms, N, C)
            alpha_features = self.fourier_dict.get_atoms(x.detach(), self.indices_alpha).detach()
            alpha_features = alpha_features.permute(1, 0, 2)  # (N, n_atoms, C)
        sel = self.feature_selector(t_emb).view(N, -1, self.C)  # (N, n_atoms, C)
        mask = torch.softmax(sel, dim=1)  # (N, n_atoms, C), softmax over the n_atoms dimension
        alpha = torch.sum(mask * alpha_features, dim=1)  # (N, C), weighted sum of atom features

        # --- Beta: NeRF-encoded Fourier Feature Network ---
        # Multi-scale features: many projections at low freqs, fewer at high freqs.
        nerf_feats = self.nerf_features(x)                               # (N, nerf_out_dim)
        spatial_in = torch.cat([x, nerf_feats], dim=-1)                  # (N, d + nerf_out_dim)

        beta = self.ff_beta(torch.cat([t_emb, spatial_in], dim=-1))     # (N, C)

        return alpha, beta