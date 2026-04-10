import itertools
import math

from .base import NeuralField, FourierFeatures, RMSNorm, _init_orthogonal
from .time_embedding import SinusoidalTimeEmbedding

import torch
import torch.nn.functional as F
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


class NewTimeEvolvingField(NeuralField):
    """OldTimeEvolvingField + separable Fourier branch, mixed via a learnable angle.

        v(t, x) = cos(θ) · v_mlp(t, x)  +  sin(θ) · v_fourier(t, x)

    v_mlp     — identical to OldTimeEvolvingField: BN(spatial_mlp(FF(x))) @ W(t).
                E_x[||v_mlp||²] ≈ 1 at every t.

    v_fourier — separable Fourier products over the integer lattice:
                Σ_{k,m} Ã_{k,m,c} · ∏_i (a·sin(2πf_{m,i}x_i) + b·cos(2πf_{m,i}x_i))
                Per-dim (a,b) unit-normalised → covers ALL 2^d axis-aligned atom types.
                Parseval-normalised: ||v_fourier_c||_L2 ≈ 1.
                Sliding window sweeps active frequency band from low_freq to high_freq.

    θ is a learnable scalar initialised at π/4 (equal mix).
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        # ── MLP branch (identical to OldTimeEvolvingField) ──
        spatial_hidden_dims: tuple = (256, 256, 256, 256),
        time_hidden_dims: tuple = (256, 256),
        feature_dim: int = 256,
        use_fourier_features: bool = True,
        n_fourier_features: int = 64,
        fourier_sigma: float = 10.0,
        n_time_freqs: int = 8,
        activation=nn.ReLU,
        post_combination_dims: tuple = (),
        # ── Fourier branch ──
        n_modes: int = 64,
        low_freq: float = 0.0,
        high_freq: float = 32.0,
        bandwidth: float = 6.0,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        D, C = feature_dim, output_dim
        self._D          = D
        self._C          = C
        self._n_fourier  = n_fourier_features
        self._coords_dim = coords_dim

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        D_t = self.time_embedding.out_dim

        # ── v_mlp: spatial MLP × Frobenius-normalised time matrix ─────────────
        if use_fourier_features:
            self.ff    = FourierFeatures(coords_dim, n_fourier_features, fourier_sigma)
            spatial_in = 2 * n_fourier_features + coords_dim
        else:
            self.ff    = None
            spatial_in = coords_dim

        self.skip = nn.Linear(spatial_in, C, bias=False)
        nn.init.zeros_(self.skip.weight)

        spatial_layers = []
        prev = spatial_in
        for h in spatial_hidden_dims:
            spatial_layers += [nn.Linear(prev, h), activation()]
            prev = h
        spatial_layers.append(nn.Linear(prev, D))
        self.spatial_mlp = nn.Sequential(*spatial_layers)
        self.spatial_mlp.apply(_init_orthogonal)
        self.spatial_bn = nn.BatchNorm1d(D, affine=False)

        time_layers = []
        prev = D_t
        for h in time_hidden_dims:
            time_layers += [nn.Linear(prev, h), activation()]
            prev = h
        time_layers.append(nn.Linear(prev, D * C))
        self.time_mlp = nn.Sequential(*time_layers)
        self.time_mlp.apply(_init_orthogonal)

        if post_combination_dims:
            layers = []
            prev = C
            for h in post_combination_dims:
                lin = nn.Linear(prev, h)
                nn.init.orthogonal_(lin.weight); nn.init.zeros_(lin.bias)
                layers += [lin, activation()]
                prev = h
            last = nn.Linear(prev, C)
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
            layers.append(last)
            self.post_combination = nn.Sequential(*layers)
        else:
            self.post_combination = None

        # ── v_fourier: per-atom-type amplitude control ────────────────────────
        self._n_modes   = n_modes
        self._low_freq  = low_freq
        self._high_freq = high_freq
        self._bandwidth = bandwidth
        d               = coords_dim
        J               = 2 ** d   # one amplitude per atom type (sin/cos per dim)

        n_fourier_out = n_modes * J * output_dim
        flayers = []
        prev = D_t
        for h in time_hidden_dims:
            flayers += [nn.Linear(prev, h), activation()]
            prev = h
        last = nn.Linear(prev, n_fourier_out)
        _init_orthogonal(last)
        flayers.append(last)
        self.fourier_time_mlp = nn.Sequential(*flayers)
        self.fourier_time_mlp[:-1].apply(_init_orthogonal)

        freq_vecs = self._make_freq_grid(d, n_modes)            # (M, d)
        self.register_buffer("_freq_vecs", freq_vecs,                   persistent=False)
        self.register_buffer("_freq_mags", freq_vecs.norm(dim=-1),      persistent=False)

        # _atom_sc[j, i] = True means use cos for dim i in atom type j, False = sin
        atom_sc = torch.tensor(
            [[bool(j >> i & 1) for i in range(d)] for j in range(J)]
        )                                                        # (J, d)
        self.register_buffer("_atom_sc", atom_sc, persistent=False)

        # Learnable energy-preserving mix angle (π/4 = equal mix at init)
        self.mix_theta = nn.Parameter(torch.tensor(math.pi / 4.0))

    @staticmethod
    def _make_freq_grid(d: int, n_modes: int) -> torch.Tensor:
        K = int(math.ceil(n_modes ** (1.0 / d))) + 2
        freqs = torch.tensor(
            list(itertools.product(range(-K, K + 1), repeat=d)), dtype=torch.float32
        )
        mags  = freqs.norm(dim=-1)
        freqs = freqs[mags > 0]
        mags  = mags[mags > 0]
        freqs = freqs[mags.argsort(stable=True)]
        if freqs.shape[0] >= n_modes:
            return freqs[:n_modes]
        return freqs.repeat((n_modes + freqs.shape[0] - 1) // freqs.shape[0], 1)[:n_modes]

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        C = self._C
        M = self._n_modes
        d = self._coords_dim
        J = 2 ** d

        t_emb = self.time_embedding(t)                              # (N, D_t)

        # ── v_mlp ──────────────────────────────────────────────────────────────
        if self.ff is not None:
            ff_out = self.ff(x)                                      # (N, 2K)
            x_in   = torch.cat([ff_out, x], dim=-1)                  # (N, spatial_in)
        else:
            x_in = x

        g     = self.spatial_bn(self.spatial_mlp(x_in))            # (N, D)
        W_raw = self.time_mlp(t_emb).reshape(N, self._D, C)
        W     = W_raw / (W_raw.pow(2).sum(dim=(-2,-1), keepdim=True).sqrt() + 1e-8)
        v_mlp = einsum('nd,ndc->nc', g, W) + self.skip(x_in)
        if self.post_combination is not None:
            v_mlp = v_mlp + self.post_combination(v_mlp)

        # ── v_fourier ──────────────────────────────────────────────────────────
        amps = self.fourier_time_mlp(t_emb).reshape(N, M, J, C)    # (N, M, J, C)

        freq_mags   = self._freq_mags.to(dtype=x.dtype)
        c0          = max(self._low_freq  + self._bandwidth / 2.0, freq_mags.min().item())
        c1          = min(self._high_freq - self._bandwidth / 2.0, freq_mags.max().item())
        target_freq = c0 + t.view(N, 1).clamp(0, 1) * (c1 - c0)   # (N, 1)
        freq_w      = torch.exp(-0.5 * ((freq_mags.view(1, M) - target_freq) / (self._bandwidth / 3.0)) ** 2)
        freq_w      = freq_w / (freq_w.sum(dim=-1, keepdim=True) + 1e-8)
        amps        = amps * freq_w[:, :, None, None]               # (N, M, J, C)

        parseval_norm = ((0.5 ** d) * amps.pow(2).sum(dim=(1, 2))).sqrt().clamp(min=1e-8)
        amps        = amps / parseval_norm[:, None, None, :]        # (N, M, J, C)

        freq_vecs   = self._freq_vecs.to(dtype=x.dtype)
        base_angles = 2.0 * math.pi * x[:, None, :] * freq_vecs[None, :, :]  # (N, M, d)
        S           = torch.sin(base_angles)                        # (N, M, d)
        Cv          = torch.cos(base_angles)                        # (N, M, d)
        # phi[n, m, j] = ∏_i { cos if _atom_sc[j,i] else sin }(2π f_{m,i} x_i)
        sc          = torch.where(
            self._atom_sc[None, None, :, :],                        # (1, 1, J, d)
            Cv[:, :, None, :], S[:, :, None, :]                     # (N, M, 1, d)
        )                                                            # (N, M, J, d)
        phi         = sc.prod(dim=-1)                               # (N, M, J)
        v_fourier   = (amps * phi.unsqueeze(-1)).sum(dim=(1, 2))    # (N, C)

        v_mlp_rms = v_mlp.pow(2).mean().sqrt().clamp(min=1e-8)
        v_mlp = v_mlp / v_mlp_rms

        return torch.cos(self.mix_theta) * v_mlp + torch.sin(self.mix_theta) * v_fourier


class FullSpectrumTimeEvolvingField(NeuralField):
    """v(t, x) = Σ_{m,j} Â_{m,j,c}(t) · φ_{m,j}(x)

    All M frequency modes in (low_freq, high_freq] are active at every Householder
    step — no sliding window.  The training objective determines which frequencies
    are modified; full spectral coverage is available at all times.

    φ_{m,j}(x) = ∏_{i=1}^d { sin(2π f_{m,i} x_i)  if bit i of j = 0,
                               cos(2π f_{m,i} x_i)  if bit i of j = 1 }

    The 2^d atom types j ∈ {0,...,2^d-1} cover every axis-aligned sin/cos
    combination at each integer frequency vector f_m ∈ ℤ^d.

    A single time MLP maps τ(t) → A_{m,j,c}(t) for all (m, j, c).
    Parseval normalisation enforces ||v_c(t,·)||_L2 ≈ 1 at every t.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_modes: int = 128,
        low_freq: float = 0.0,
        high_freq: float = 32.0,
        time_hidden_dims: tuple = (256, 256),
        n_time_freqs: int = 8,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        d, C = coords_dim, output_dim
        J    = 2 ** d

        self._coords_dim = d
        self._C          = C

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        D_t = self.time_embedding.out_dim

        # Frequency grid first — actual count may be < n_modes if the range is small.
        freq_vecs = self._make_freq_grid(d, n_modes, low_freq, high_freq)
        self._n_modes = freq_vecs.shape[0]
        self.register_buffer("_freq_vecs", freq_vecs, persistent=False)

        # Time MLP: τ(t) → A_{m,j,c} for every (mode, atom-type, channel)
        layers = []
        prev = D_t
        for h in time_hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        last = nn.Linear(prev, self._n_modes * J * C)
        _init_orthogonal(last)
        layers.append(last)
        self.time_mlp = nn.Sequential(*layers)
        self.time_mlp[:-1].apply(_init_orthogonal)

        # _atom_sc[j, i] = True  →  use cos for spatial dim i in atom type j
        atom_sc = torch.tensor(
            [[bool(j >> i & 1) for i in range(d)] for j in range(J)]
        )                                                        # (J, d)
        self.register_buffer("_atom_sc", atom_sc, persistent=False)

    @staticmethod
    def _make_freq_grid(d: int, n_modes: int, low_freq: float, high_freq: float) -> torch.Tensor:
        # Enumerate all integer vectors up to ceil(high_freq)+1 in each dim
        K = max(int(math.ceil(high_freq)) + 1, int(math.ceil(n_modes ** (1.0 / d))) + 2)
        freqs = torch.tensor(
            list(itertools.product(range(-K, K + 1), repeat=d)), dtype=torch.float32
        )
        mags  = freqs.norm(dim=-1)
        mask  = (mags > low_freq) & (mags <= high_freq)
        freqs, mags = freqs[mask], mags[mask]
        freqs = freqs[mags.argsort(stable=True)]
        return freqs[:n_modes]

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        C = self._C
        d = self._coords_dim
        M = self._n_modes
        J = 2 ** d

        t_emb = self.time_embedding(t)                              # (N, D_t)

        # Time-varying amplitudes for all modes and atom types
        amps  = self.time_mlp(t_emb).reshape(N, M, J, C)           # (N, M, J, C)

        # Parseval normalisation: ||v_c||_L2 ≈ 1
        parseval_norm = ((0.5 ** d) * amps.pow(2).sum(dim=(1, 2))).sqrt().clamp(min=1e-8)
        amps  = amps / parseval_norm[:, None, None, :]

        # Separable-product basis functions
        freq_vecs   = self._freq_vecs.to(dtype=x.dtype)
        base_angles = 2.0 * math.pi * x[:, None, :] * freq_vecs[None, :, :]  # (N, M, d)
        S           = torch.sin(base_angles)                        # (N, M, d)
        Cv          = torch.cos(base_angles)                        # (N, M, d)
        sc          = torch.where(
            self._atom_sc[None, None, :, :],                        # (1, 1, J, d)
            Cv[:, :, None, :], S[:, :, None, :]                     # (N, M, 1, d)
        )                                                            # (N, M, J, d)
        phi         = sc.prod(dim=-1)                               # (N, M, J)

        return (amps * phi.unsqueeze(-1)).sum(dim=(1, 2))           # (N, C)
