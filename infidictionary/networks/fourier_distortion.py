import math
import torch
from torch import nn

from .base import TimeEvolvingField, RMSNorm, _build_mlp
from .time_embedding import SinusoidalTimeEmbedding

class FourierDistortedField(TimeEvolvingField):
    """Frequency-sweeping Fourier field with a fixed random direction bank.

    Compared to ``FourierDistortedFieldV2``, all time-dependent MLPs that
    produced directions, phases and feature masks are replaced by fixed random
    buffers sampled once at construction time.  The only time-dependent
    computation is the scalar frequency schedule ``_compute_freq``, which sweeps
    the spatial frequency from ``low_freq`` to ``high_freq`` as t increases.

    At each point (t_n, x_n) the two output streams are::

        # fixed buffers
        W_orth   ∈ R^{n_modes x K x d}   (unit-norm directions)
        phases   ∈ R^{n_modes x K}        (uniform random in [0, 1))

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

        W = torch.randn(n_modes, self.K, coords_dim)
        W = W / W.norm(dim=-1, keepdim=True)         # unit-norm directions
        self.register_buffer('W_orth', W)
        self.register_buffer('phases', torch.rand(n_modes, self.K))
            
        self.alpha_ff = _build_mlp(
            emb_dim + self.n_modes * self.K,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )
        self.beta_ff = _build_mlp(
            emb_dim + self.n_modes * self.K,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )
    
    @staticmethod
    def _build_mlp(
        in_dim: int, out_dim: int, hidden_dims: tuple, activation, 
        use_batchnorm: bool, use_rmsnorm: bool, bias: bool,
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

    def _compute_freq(self, t: torch.Tensor):
        t_scaled = ((t.clamp(self.low_t, self.high_t) - self.low_t)
                    / (self.high_t - self.low_t + 1e-3)).unsqueeze(-1)
        t_scaled = (t_scaled
                    + torch.arange(self.freq_strides, device=t.device).unsqueeze(0)
                    / self.freq_strides)
        t_scaled = t_scaled % 1.0
        if self.gamma < 0:
            freq_scale = 1 - (1 - t_scaled) ** (-self.gamma)
        else:
            freq_scale = t_scaled ** self.gamma
        freq_scale = torch.cat([freq_scale, 1 - freq_scale], dim=-1)  # (N, K)
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
        phases1 = self.phases[None, :, :K_half]                        # (n_modes, K/2)

        arg1    = 2 * math.pi * freq1[:, None, :] * (proj1 + phases1)
        feat1   = torch.cat([torch.sin(arg1), torch.cos(arg1)], dim=-1)  # (N, n_modes, K)
        feat1   = feat1.reshape(N, self.n_modes * self.K)

        # Stream 2: second K/2 slots, fine→coarse (complementary) schedule
        proj2   = proj[:, :, K_half:]                            # (N, n_modes, K/2)
        phases2 = self.phases[None, :, K_half:]                        # (n_modes, K/2)
        arg2    = 2 * math.pi * freq2[:, None, :] * (proj2 + phases2)
        feat2   = torch.cat([torch.sin(arg2), torch.cos(arg2)], dim=-1)  # (N, n_modes, K)
        feat2   = feat2.reshape(N, self.n_modes * self.K)

        alpha = self.alpha_ff(torch.cat([t_emb, feat1], dim=-1))
        beta  = self.beta_ff( torch.cat([t_emb, feat2], dim=-1))

        return alpha, beta

