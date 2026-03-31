import math

import torch
from torch import nn

from .base import NeuralField, FourierFeatures, RMSNorm
from .time_embedding import SinusoidalTimeEmbedding


class FactoredTimeEvolvingField(NeuralField):
    """v(t, x) = post_mlp( (RMSNorm(spatial_mlp_FiLM(FF(x), t)) @ W) / sqrt(C) ) + b(t).

    Optimized Version:
      - spatial_mlp: Heavily conditioned on `t` via FiLM. Ends with RMSNorm (||g||^2 = D).
      - spatial_proj: A STATIC orthogonal projection W. Removes the O(N*D*C) HyperNetwork
                      bottleneck. Scaled by 1/sqrt(C) to ensure E[||v||^2] ≈ 1.
      - dc_bias_mlp: Adds b(t) to handle the (0,0) Fourier atom.
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
        n_mixture_features: int = 0,
        mixture_min_freq: float = 0.5,
        mixture_max_freq: float = 8.0,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        D, C = feature_dim, output_dim
        self._D = D
        self._C = C
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        # ── Spatial MLP (ModuleList for FiLM injection) ──────────────────────
        if use_fourier_features:
            self.ff = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
            spatial_in = 2 * n_fourier_features + coords_dim  # FF(x) ++ x
        else:
            self.ff = None
            spatial_in = coords_dim

        self._spatial_hidden_dims = [int(h) for h in spatial_hidden_dims]
        self.spatial_linears = nn.ModuleList()
        self.spatial_acts    = nn.ModuleList()
        prev = spatial_in
        self.spatial_features_skip = nn.Linear(prev, C, bias=False)
        for h in self._spatial_hidden_dims:
            lin = nn.Linear(prev, h)
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.spatial_linears.append(lin)
            self.spatial_acts.append(activation())
            prev = h

        self.spatial_final = nn.Linear(prev, D)
        nn.init.orthogonal_(self.spatial_final.weight)
        nn.init.zeros_(self.spatial_final.bias)
        self.spatial_rms = RMSNorm()

        # ── REFACTORED: Static Projection Matrix W ───────────────────────────
        # Replaces the computationally heavy dynamic W(t) generation.
        # Bias is False because dc_bias_mlp handles the output offset.
        self.spatial_proj = nn.Linear(D, C, bias=False)
        nn.init.orthogonal_(self.spatial_proj.weight)
        # nn.init.zeros_(self.spatial_proj.weight) # TODO: ablate with this

        # ── FiLM MLP: (δγ_l, β_l) for each spatial hidden layer ─────────────
        time_in = self.time_embedding.out_dim
        film_out_dim = sum(2 * h for h in self._spatial_hidden_dims)
        film_layers: list = []
        prev_f = time_in
        for h in time_hidden_dims:
            lin = nn.Linear(prev_f, int(h))
            nn.init.orthogonal_(lin.weight)
            nn.init.zeros_(lin.bias)
            film_layers += [lin, activation()]
            prev_f = int(h)
        last_film = nn.Linear(prev_f, film_out_dim)
        nn.init.zeros_(last_film.weight)
        nn.init.zeros_(last_film.bias)
        film_layers += [last_film]
        self.film_mlp = nn.Sequential(*film_layers)

        # ── DC bias: b(t) ∈ ℝᶜ ──────────────────────────────────────────────────
        bias_layers: list = []
        prev_b = time_in
        if time_hidden_dims:
            h_b = int(time_hidden_dims[0])
            lin_b = nn.Linear(prev_b, h_b)
            nn.init.orthogonal_(lin_b.weight)
            nn.init.zeros_(lin_b.bias)
            bias_layers += [lin_b, activation()]
            prev_b = h_b
        last_b = nn.Linear(prev_b, C)
        nn.init.zeros_(last_b.weight)
        # nn.init.constant_(last_b.bias, 0.01) # TODO: ablate with this
        nn.init.zeros_(last_b.bias)
        bias_layers += [last_b]
        self.dc_bias_mlp = nn.Sequential(*bias_layers)

        # ── Post-combination nonlinear MLP ───────────────────────────────────
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
        x_in = torch.cat([self.ff(x), x], dim=-1) if self.ff is not None else x

        t_emb = self.time_embedding(t)
        film_all = self.film_mlp(t_emb)

        # Spatial MLP with per-layer FiLM conditioning
        h = x_in
        offset = 0
        for linear, act, h_dim in zip(self.spatial_linears, self.spatial_acts, self._spatial_hidden_dims):
            h = act(linear(h))
            delta_gamma = film_all[:, offset : offset + h_dim]
            beta        = film_all[:, offset + h_dim : offset + 2 * h_dim]
            h = (1 + delta_gamma) * h + beta
            offset += 2 * h_dim

        g = self.spatial_rms(self.spatial_final(h))                              # (N, D)

        v = self.spatial_proj(g) / math.sqrt(self._C) + self.spatial_features_skip(x_in)                            # (N, C)

        if self.post_combination is not None:
            v = v + self.post_combination(v)

        v = v + self.dc_bias_mlp(t_emb)                                          # (N, C)

        return v
