import math

import torch
from torch import nn

from .base import NeuralField
from .time_embedding import SinusoidalTimeEmbedding


class FourierMixtureField(NeuralField):
    """v(t, x) = (W_0 + ΔW(t)) @ [sin(2π B(t)ᵀx), cos(2π B(t)ᵀx), x] + b(t)

    Both B(t) and W(t) are time-dependent, so the function space is genuinely
    different at each t — not just a scaling of fixed spatial features.

        B(t) = B_0 + ΔB(t)   (d, n)   log-uniform B_0, zero-init ΔB
        W(t) = W_0 + ΔW(t)   (C, ff_dim)  orthogonal W_0, zero-init ΔW
        b(t) ∈ R^C            zero-init DC bias

    Time MLP output size: d*n + C*(2n+d) + C.
    With d=2, n=64, C=2: 128 + 2*130 + 2 = 390 — very small.

    Args:
        coords_dim:         Spatial dimension d.
        output_dim:         Output channels C.
        n_fourier_features: Number of frequency vectors n; FF dim = 2n+d.
        min_freq:           Log-uniform lower bound for B_0.
        max_freq:           Log-uniform upper bound for B_0.
        n_time_freqs:       Frequencies for sinusoidal time embedding.
        time_hidden_dims:   Hidden widths for the time MLP.
        activation:         Activation constructor.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_fourier_features: int = 64,
        min_freq: float = 0.5,
        max_freq: float = 8.0,
        n_time_freqs: int = 16,
        time_hidden_dims: tuple = (256, 256),
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        d, C = coords_dim, output_dim
        self._C = C
        self._d = d
        n = n_fourier_features
        self._n = n
        ff_dim = 2 * n + d
        self._ff_dim = ff_dim

        # ── Log-uniform base frequencies B_0 ─────────────────────────────────
        direction = torch.randn(d, n)
        direction = direction / direction.norm(dim=0, keepdim=True).clamp(min=1e-8)
        log_freqs = torch.rand(n) * math.log(max_freq / min_freq) + math.log(min_freq)
        B0 = direction * log_freqs.exp().unsqueeze(0)          # (d, n)
        self.register_buffer("B0", B0)

        W0 = torch.empty(C, ff_dim)
        nn.init.orthogonal_(W0)
        self.register_buffer("W0", W0)

        # ── Time MLP: t_emb → (ΔB, ΔW, b) ───────────────────────────────────
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        time_in = self.time_embedding.out_dim
        time_out = d * n + C * ff_dim + C

        layers: list = []
        prev = time_in
        for h in time_hidden_dims:
            lin = nn.Linear(prev, int(h))
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, activation()]
            prev = int(h)
        last = nn.Linear(prev, time_out)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        layers += [last]
        self.time_mlp = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)  x: (N, d)
        N = x.shape[0]

        t_emb = self.time_embedding(t)
        time_out = self.time_mlp(t_emb)                               # (N, d*n + C*ff_dim + C)

        i0 = self._d * self._n
        i1 = i0 + self._C * self._ff_dim
        delta_B = time_out[:, :i0].view(N, self._d, self._n)          # (N, d, n)
        delta_W = time_out[:, i0:i1].view(N, self._C, self._ff_dim)   # (N, C, ff_dim)
        b       = time_out[:, i1:]                                     # (N, C)

        B = self.B0.unsqueeze(0) + delta_B                             # (N, d, n)
        proj = 2 * math.pi * (x.unsqueeze(2) * B).sum(dim=1)          # (N, n)
        ff = torch.cat([torch.sin(proj), torch.cos(proj), x], dim=-1)  # (N, ff_dim)

        W = self.W0.unsqueeze(0) + delta_W                             # (N, C, ff_dim)
        v = torch.einsum("ncf, nf -> nc", W, ff) + b                   # (N, C)
        return v
