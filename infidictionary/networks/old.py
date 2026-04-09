from .base import NeuralField, FourierFeatures, RMSNorm, _init_orthogonal
from .time_embedding import SinusoidalTimeEmbedding

import torch
from torch import einsum, nn

class OldTimeEvolvingField(NeuralField):
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
        adaptive_modulation:   if True, the Fourier frequency matrix B becomes
                               time-dependent: B(t) = B_fixed + freq_mlp(t_emb),
                               so the network can tune which frequencies it attends
                               to at each time.  Ignored when use_fourier_features
                               is False.  Default: False (original behaviour).
        freq_hidden_dims:      hidden widths for the frequency-modulation MLP;
                               only used when adaptive_modulation=True.
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
        adaptive_modulation: bool = False,
        freq_hidden_dims: tuple = (256,),
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        D, C = feature_dim, output_dim
        self._D = D
        self._C = C
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        self._n_fourier = n_fourier_features
        self._coords_dim = coords_dim

        # ── Spatial MLP + BN ─────────────────────────────────────────────────
        if use_fourier_features:
            self.ff = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
            spatial_in = 2 * n_fourier_features + coords_dim  # FF(x) ++ x
        else:
            self.ff = None
            adaptive_modulation = False  # no FF → nothing to modulate
            spatial_in = coords_dim

        self.skip = nn.Linear(spatial_in, C, bias=False)
        nn.init.zeros_(self.skip.weight)  # zero-init so it starts as identity residual


        # ── Frequency-modulation MLP: t_emb → ΔB(t) ∈ ℝᵈˣᴷ ─────────────────
        # B(t) = B_fixed + ΔB(t).  Last layer zero-inited so ΔB(0) ≈ 0 and
        # the network starts identical to the fixed-FF baseline.
        self.adaptive_modulation = adaptive_modulation
        if adaptive_modulation:
            freq_layers = []
            prev = self.time_embedding.out_dim
            for h in freq_hidden_dims:
                freq_layers += [nn.Linear(prev, h), activation()]
                prev = h
            last = nn.Linear(prev, coords_dim * n_fourier_features)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
            freq_layers += [last]
            self.freq_mlp = nn.Sequential(*freq_layers)
            self.freq_mlp[:-1].apply(_init_orthogonal)  # hidden layers only
        else:
            self.freq_mlp = None

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
        t_emb = self.time_embedding(t)                                      # (N, D_t)

        if self.ff is not None:
            if self.adaptive_modulation:
                # B(t) = B_fixed + ΔB(t),  shape (N, d, K)
                delta_B = self.freq_mlp(t_emb).reshape(-1, self._coords_dim, self._n_fourier)
                B_t = self.ff.B.unsqueeze(0) + delta_B                     # (N, d, K)
                proj = 2 * torch.pi * torch.einsum('nd,ndk->nk', x, B_t)  # (N, K)
                ff_out = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (N, 2K)
            else:
                ff_out = self.ff(x)                                         # (N, 2K)
            x_in = torch.cat([ff_out, x], dim=-1)                          # (N, spatial_in)
        else:
            x_in = x
        g = self.spatial_bn(self.spatial_mlp(x_in))                        # (N, D)

        W_raw = self.time_mlp(t_emb).reshape(-1, self._D, self._C)         # (N, D, C)
        W = W_raw / (W_raw.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)

        
        v = einsum('nd,ndc->nc', g, W) + self.skip(x_in)       # (N, C)

        if self.post_combination is not None:
            v = v + self.post_combination(v)

        return v
