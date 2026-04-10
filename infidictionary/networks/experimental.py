import itertools
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from .base import NeuralField, FourierFeatures, RMSNorm, _init_orthogonal
from .time_embedding import SinusoidalTimeEmbedding

# ── V1: Sparse softmax mode selector (fragile to init_logit_scale / n_components) ──
#
from .base import NeuralField, _init_orthogonal
from .time_embedding import SinusoidalTimeEmbedding

import torch
import torch.nn.functional as F
from torch import nn
import math

class FourierTimeEvolvingField(NeuralField):
    """Separable Fourier field for arbitrary spatial dimension d.

    v_c(t, x) = Σ_{k,m}  Ã_{k,m,c}(t) · ∏_{i=1}^d [ a_{k,m,i,c}·sin(2π·f_{m,i}·x_i)
                                                       + b_{k,m,i,c}·cos(2π·f_{m,i}·x_i) ]

    f_m = (f_{m,1},...,f_{m,d}) are fixed integer frequency vectors from the Fourier
    lattice (sorted by L2 magnitude, DC excluded).

    Per-dimension factors a·sin + b·cos (with (a,b) unit-normalised) cover ALL types
    of axis-aligned Fourier atoms simultaneously:
      d=1: sin and cos
      d=2: sin·sin, sin·cos, cos·sin, cos·cos
      d=k: all 2^k combinations

    Parseval norm (after per-dim unit normalisation):
      ||∏_i (a_i·sin + b_i·cos)||²_L2 = (1/2)^d   (factors are independent)
    so  ||v_c||²_L2 ≈ Σ_{k,m} A_{k,m,c}² · (1/2)^d = 1  after amplitude normalisation.

    Sliding window sweeps active frequency magnitude from low_freq to high_freq
    as t goes from 0 to 1.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_modes: int = 32,
        n_components: int = 2,
        time_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        low_freq: float = 1.0,
        high_freq: float = 32.0,
        bandwidth: float = 6.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        assert low_freq + bandwidth <= high_freq, "bandwidth too large for the given freq bounds"

        self.n_modes = n_modes
        self.n_components = n_components
        self.C = output_dim
        self.d = coords_dim
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.bandwidth = bandwidth

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        # MLP output layout per sample: (N, K, 1 + 2d, M, C)
        #   [:, :,    0,     :, :] = amplitude A    → (N, K, M, C)
        #   [:, :,  1:1+d,   :, :] = sin-weights a  → (N, K, d, M, C)
        #   [:, :, 1+d:1+2d, :, :] = cos-weights b  → (N, K, d, M, C)
        self._n_params = n_components * (1 + 2 * coords_dim) * n_modes * output_dim

        layers = []
        prev = self.time_embedding.out_dim
        for h in time_hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        last = nn.Linear(prev, self._n_params)
        _init_orthogonal(last)
        layers.append(last)
        self.time_mlp = nn.Sequential(*layers)
        self.time_mlp[:-1].apply(_init_orthogonal)

        # Integer frequency lattice: (M, d), sorted by L2 magnitude, DC excluded.
        freq_vecs = self._make_freq_grid(coords_dim, n_modes)   # (M, d)
        freq_mags = freq_vecs.norm(dim=-1)                       # (M,)
        self.register_buffer("_freq_vecs", freq_vecs, persistent=False)
        self.register_buffer("_freq_mags", freq_mags, persistent=False)

    @staticmethod
    def _make_freq_grid(d: int, n_modes: int) -> torch.Tensor:
        """Return (n_modes, d) integer frequency vectors sorted by L2 norm (DC excluded)."""
        K = int(math.ceil(n_modes ** (1.0 / d))) + 2
        all_freqs = list(itertools.product(range(-K, K + 1), repeat=d))
        freqs = torch.tensor(all_freqs, dtype=torch.float32)   # ((2K+1)^d, d)
        mags  = freqs.norm(dim=-1)
        freqs = freqs[mags > 0]
        mags  = mags[mags > 0]
        order = mags.argsort(stable=True)
        freqs = freqs[order]
        if freqs.shape[0] >= n_modes:
            return freqs[:n_modes]
        repeats = (n_modes + freqs.shape[0] - 1) // freqs.shape[0]
        return freqs.repeat(repeats, 1)[:n_modes]

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[0]
        K, M, C, d = self.n_components, self.n_modes, self.C, self.d

        t_emb = self.time_embedding(t)
        raw   = self.time_mlp(t_emb)                            # (N, K*(1+2d)*M*C)

        params = raw.reshape(N, K, 1 + 2 * d, M, C)
        amps   = params[:, :,       0,     :, :]                # (N, K, M, C)
        sc_sin = params[:, :,   1:1+d,     :, :]                # (N, K, d, M, C)
        sc_cos = params[:, :, 1+d:1+2*d,   :, :]                # (N, K, d, M, C)

        # Unit-normalise (a, b) per dimension so each per-dim factor has L2 norm 1/√2
        # regardless of the MLP output scale.  This decouples amplitude from direction.
        sc_norm = (sc_sin.pow(2) + sc_cos.pow(2)).sqrt().clamp(min=1e-8)  # (N, K, d, M, C)
        sc_sin  = sc_sin / sc_norm                              # (N, K, d, M, C)
        sc_cos  = sc_cos / sc_norm

        # Sliding frequency window on mode magnitudes.
        # Clamp the sweep range to [grid_min, grid_max] so the window never
        # drifts outside the actual frequency grid (which would zero-out v).
        freq_mags   = self._freq_mags.to(dtype=x.dtype)         # (M,)
        grid_min    = freq_mags.min()
        grid_max    = freq_mags.max()

        t_scaled    = t.view(N, 1, 1).clamp(0, 1)
        c0          = self.low_freq  + self.bandwidth / 2.0
        c1          = self.high_freq - self.bandwidth / 2.0
        c0          = c0.clamp(grid_min, grid_max) if isinstance(c0, torch.Tensor) else max(c0, grid_min.item())
        c1          = c1.clamp(grid_min, grid_max) if isinstance(c1, torch.Tensor) else min(c1, grid_max.item())
        target_freq = c0 + t_scaled * (c1 - c0)                # (N, 1, 1)
        mode_dist   = freq_mags.view(1, 1, M) - target_freq     # (N, 1, M)
        sigma       = self.bandwidth / 3.0
        freq_w      = torch.exp(-0.5 * (mode_dist / sigma) ** 2)
        freq_w      = freq_w / (freq_w.sum(dim=-1, keepdim=True) + 1e-8)  # (N, 1, M)
        amps        = amps * freq_w.unsqueeze(-1)               # (N, K, M, C)

        # Parseval normalisation AFTER windowing so the active-band energy is always
        # unit norm — prevents the network from hiding amplitude in off-window modes.
        parseval_norm = ((0.5 ** d) * amps.pow(2).sum(dim=(1, 2))).sqrt().clamp(min=1e-8)
        amps = amps / parseval_norm[:, None, None, :]           # (N, K, M, C)

        # Per-dimension base angles: base[n, m, i] = 2π · f_{m,i} · x_{n,i}
        freq_vecs   = self._freq_vecs.to(dtype=x.dtype)         # (M, d)
        base_angles = 2.0 * math.pi * x[:, None, :] * freq_vecs[None, :, :]  # (N, M, d)

        # Expand for (N, K, M, d, C) broadcasting
        S = torch.sin(base_angles)[:, None, :, :, None]         # (N, 1, M, d, 1)
        Cv = torch.cos(base_angles)[:, None, :, :, None]        # (N, 1, M, d, 1)

        # Permute sc_sin/sc_cos from (N, K, d, M, C) → (N, K, M, d, C)
        a = sc_sin.permute(0, 1, 3, 2, 4)                       # (N, K, M, d, C)
        b = sc_cos.permute(0, 1, 3, 2, 4)                       # (N, K, M, d, C)

        # Per-dim factor: a·sin(2πf_i·x_i) + b·cos(2πf_i·x_i) → shape (N, K, M, d, C)
        per_dim = a * S + b * Cv                                 # (N, K, M, d, C)

        # Separable product over d, then weighted sum over K and M
        sin_prod = per_dim.prod(dim=3)                          # (N, K, M, C)
        return (amps * sin_prod).sum(dim=(1, 2))                # (N, C)
    
# NOTE: Seemingly works!
# class FourierTimeEvolvingField(NeuralField):
#     def __init__(
#         self,
#         coords_dim: int,
#         output_dim: int,
#         n_modes: int = 32,
#         n_components: int = 2,
#         time_hidden_dims: tuple = (64, 64),
#         n_time_freqs: int = 8,
#         low_freq: float = 1.0,           # New: Starting frequency bound
#         high_freq: float = 32.0,         # New: Ending frequency bound
#         bandwidth: float = 6.0,          # New: Width of the active frequency window
#         activation=nn.SiLU,
#     ):
#         super().__init__(input_dim=coords_dim, output_dim=output_dim)
#         assert coords_dim == 1
#         assert low_freq + bandwidth <= high_freq, "Bandwidth is too large for the specified frequency bounds."

#         self.n_modes = n_modes
#         self.n_components = n_components
#         self.C = output_dim
        
#         self.low_freq = low_freq
#         self.high_freq = high_freq
#         self.bandwidth = bandwidth

#         self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
#         self._n_amplitudes = n_components * 2 * n_modes * output_dim  

#         layers = []
#         prev = self.time_embedding.out_dim
#         for h in time_hidden_dims:
#             layers += [nn.Linear(prev, h), activation()]
#             prev = h
#         last = nn.Linear(prev, self._n_amplitudes)
#         _init_orthogonal(last)
#         layers.append(last)
        
#         self.time_mlp = nn.Sequential(*layers)
#         self.time_mlp[:-1].apply(_init_orthogonal)

#         # 1-based indexing for modes (1 to M)
#         m = torch.arange(1, n_modes + 1, dtype=torch.float32)
#         self.register_buffer("_modes", m, persistent=False)

#     def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         N = x.shape[0]
#         K, M, C = self.n_components, self.n_modes, self.C

#         t_emb = self.time_embedding(t)
#         raw = self.time_mlp(t_emb)

#         amps = raw.reshape(N, K, 2, M, C)
#         amp_sin = amps[:, :, 0, :, :] 
#         amp_cos = amps[:, :, 1, :, :] 

#         # ====================================================================
#         # Parameterized sliding window
#         # ====================================================================
#         t_scaled = t.view(-1, 1, 1).clamp(0, 1) 
        
#         # Calculate the center of the window at t=0 and t=1
#         c0 = self.low_freq + (self.bandwidth / 2.0)
#         c1 = self.high_freq - (self.bandwidth / 2.0)
        
#         # Interpolate the center based on time
#         target_mode = c0 + t_scaled * (c1 - c0)
        
#         # Distance of each mode from the target center
#         mode_dist = self._modes.view(1, 1, M) - target_mode
        
#         # A sigma of bandwidth/3.0 ensures the weights drop near zero 
#         # just as they hit the edges of your bandwidth window.
#         sigma = self.bandwidth / 3.0 
#         freq_w = torch.exp(-0.5 * (mode_dist / sigma) ** 2)
        
#         # Optional: If you want an ABSOLUTE hard cutoff instead of a smooth drop-off:
#         # mask = (self._modes.view(1, 1, M) >= target_mode - self.bandwidth / 2.0) & \
#         #        (self._modes.view(1, 1, M) <= target_mode + self.bandwidth / 2.0)
#         # freq_w = freq_w * mask.float()

#         freq_w = freq_w / (freq_w.sum(dim=-1, keepdim=True) + 1e-8)
#         # ====================================================================

#         freq_w_expanded = freq_w.view(-1, 1, M, 1) 
#         amp_sin = amp_sin * freq_w_expanded
#         amp_cos = amp_cos * freq_w_expanded

#         m = self._modes.to(dtype=x.dtype)
#         angles = 2.0 * math.pi * m[None, :] * x
        
#         S_all = torch.sin(angles)  
#         C_all = torch.cos(angles)  

#         S_j = torch.einsum("nkmc, nm -> nkc", amp_sin, S_all)
#         C_j = torch.einsum("nkmc, nm -> nkc", amp_cos, C_all)

#         components = S_j + C_j
#         return components.sum(dim=1)
    
    
class ExperimentalTimeEvolvingField(NeuralField):
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
                # Inductive bias: Scale base frequencies linearly by time `t`
                delta_B = self.freq_mlp(t_emb).reshape(-1, self._coords_dim, self._n_fourier)
                B_t = (self.ff.B.unsqueeze(0) * t.view(-1, 1, 1)) + delta_B       # (N, d, K)
                proj = 2 * torch.pi * torch.einsum('nd,ndk->nk', x, B_t)          # (N, K)
                ff_out = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)    # (N, 2K)
            else:
                # Inductive bias: Extract the projection logic and scale it by `t`
                proj = 2 * torch.pi * (x @ self.ff.B)
                proj = proj * t.unsqueeze(-1)                                     # (N, K)
                ff_out = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)    # (N, 2K)
                
            x_in = torch.cat([ff_out, x], dim=-1)                                 # (N, spatial_in)
        else:
            x_in = x
            
        g = self.spatial_bn(self.spatial_mlp(x_in))                               # (N, D)

        W_raw = self.time_mlp(t_emb).reshape(-1, self._D, self._C)                # (N, D, C)
        W = W_raw / (W_raw.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-8)

        v = einsum('nd,ndc->nc', g, W) + self.skip(x_in)                          # (N, C)

        if self.post_combination is not None:
            v = v + self.post_combination(v)

        return v


import torch
import torch.nn as nn
import math

class NeRFAndPWCField(NeuralField):
    """v(t, x) = w_fourier * Net_fourier + w_PWC * Net_PWC
    
    Net_fourier: NeRF-style positional encoding (powers of 2).
                 Combines sine and cosine basis functions.
    Net_PWC:     Coordinate-based hyperplanes. 
                 Random linear combinations passed through a steep tanh.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_fourier_freqs: int = 8,    # 'L' in NeRF: 2^0 up to 2^7
        n_pwc_bases: int = 128,      # Number of random hyperplanes for PWC
        time_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        pwc_steepness: float = 50.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.C = output_dim
        self.pwc_steepness = pwc_steepness

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

        # ==========================================
        # 1. FOURIER SETUP (NeRF Powers of 2)
        # ==========================================
        # Frequencies: [2^0, 2^1, 2^2, ..., 2^(L-1)]
        freqs = 2.0 ** torch.arange(n_fourier_freqs)
        self.register_buffer("fourier_freqs", freqs, persistent=False)
        
        # Total Fourier modes = dimensions * number of frequencies
        self.M_f = coords_dim * n_fourier_freqs 

        # ==========================================
        # 2. PWC SETUP (Random Hyperplanes)
        # ==========================================
        self.M_p = n_pwc_bases
        
        # Wx + b random projections. 
        # Biases initialized uniformly to ensure planes intersect the data domain.
        W_pwc = torch.randn(n_pwc_bases, coords_dim)
        b_pwc = torch.empty(n_pwc_bases).uniform_(-math.pi, math.pi)
        
        self.register_buffer("W_pwc", W_pwc, persistent=False)
        self.register_buffer("b_pwc", b_pwc, persistent=False)

        # ==========================================
        # 3. PHASE MLPs (Time -> Coefficients)
        # ==========================================
        def build_phase_mlp(out_features):
            layers = []
            prev = self.time_embedding.out_dim
            for h in time_hidden_dims:
                layers += [nn.Linear(prev, h), activation()]
                prev = h
            last = nn.Linear(prev, out_features)
            _init_orthogonal(last)
            layers.append(last)
            
            mlp = nn.Sequential(*layers)
            mlp[:-1].apply(_init_orthogonal)
            return mlp

        # Fourier needs 2 values (cos_phi, sin_phi) per mode per channel
        self.phase_mlp_fourier = build_phase_mlp(out_features = self.M_f * self.C * 2)
        
        # PWC only needs 1 weight per hyperplane per channel
        self.phase_mlp_pwc = build_phase_mlp(out_features = self.M_p * self.C)

        # Energy preserving mixing angle
        self.mix_theta = nn.Parameter(torch.tensor([math.pi / 4]))

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)   x: (N, D)
        N, C = x.shape[0], self.C
        t_emb = self.time_embedding(t)

        # ---------------------------------------------------------
        # BRANCH A: Fourier (Multi-resolution Smooth Details)
        # ---------------------------------------------------------
        # 1. Project x into NeRF frequencies: [N, D] -> [N, D, L] -> [N, M_f]
        x_expanded = x.unsqueeze(-1) * self.fourier_freqs
        x_fourier = x_expanded.reshape(N, self.M_f)
        
        angles = 2.0 * math.pi * x_fourier
        S_f = torch.sin(angles)   # (N, M_f)
        C_f = torch.cos(angles)   # (N, M_f)

        # 2. Get time-varying coefficients
        phases_f = self.phase_mlp_fourier(t_emb).reshape(N, self.M_f, C, 2)
        cos_phi_f = phases_f[..., 0]  # (N, M_f, C)
        sin_phi_f = phases_f[..., 1]  # (N, M_f, C)

        # 3. Combine
        v_fourier = (S_f[:, :, None] * cos_phi_f + C_f[:, :, None] * sin_phi_f).sum(dim=1)
        v_fourier = v_fourier / math.sqrt(self.M_f)


        # ---------------------------------------------------------
        # BRANCH B: PWC (Coordinate-based Sharp Polytopes)
        # ---------------------------------------------------------
        # 1. Random linear projection: Wx + b -> (N, M_p)
        preact = x @ self.W_pwc.T + self.b_pwc
        
        # 2. Apply steep activation to create structural step functions
        basis_pwc = torch.sign(self.pwc_steepness * preact)

        # 3. Get time-varying weights
        weights_pwc = self.phase_mlp_pwc(t_emb).reshape(N, self.M_p, C)

        # 4. Combine (No sines/cosines needed here)
        v_pwc = (basis_pwc[:, :, None] * weights_pwc).sum(dim=1)
        v_pwc = v_pwc / math.sqrt(self.M_p)


        # ---------------------------------------------------------
        # FINAL MIXING
        # ---------------------------------------------------------
        w_fourier = torch.cos(self.mix_theta)
        w_pwc = torch.sin(self.mix_theta)

        return (w_fourier * v_fourier) + (w_pwc * v_pwc)
    
# class MixedFourierPWCField(NeuralField):
#     """v(t, x) = w_fourier * Net_fourier + w_PWC * Net_PWC
    
#     Net_fourier uses continuous sine/cosine bases.
#     Net_PWC uses a differentiable soft-square-wave basis via high-gain tanh.
    
#     Args:
#         coords_dim:       Spatial dimension D (e.g., 1, 2, 3).
#         output_dim:       Output channels C.
#         n_modes:          Number of fixed modes M.
#         time_hidden_dims: Hidden widths of the phase MLPs.
#         n_time_freqs:     Sinusoidal time-embedding frequencies.
#         pwc_steepness:    Gain factor for the tanh approximation. Higher = sharper edges.
#         freq_scale:       Standard deviation for the random frequency matrix B.
#         activation:       Activation constructor.
#     """

#     def __init__(
#         self,
#         coords_dim: int,
#         output_dim: int,
#         n_modes: int = 16,
#         time_hidden_dims: tuple = (64, 64),
#         n_time_freqs: int = 8,
#         pwc_steepness: float = 50.0,
#         freq_scale: float = 1.0,
#         activation=nn.SiLU,
#     ):
#         super().__init__(input_dim=coords_dim, output_dim=output_dim)

#         self.n_modes = n_modes
#         self.C = output_dim
#         self.pwc_steepness = pwc_steepness

#         self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)

#         n_out = n_modes * output_dim * 2

#         # Helper to build identical independent MLPs
#         def build_phase_mlp():
#             layers = []
#             prev = self.time_embedding.out_dim
#             for h in time_hidden_dims:
#                 layers += [nn.Linear(prev, h), activation()]
#                 prev = h
#             last = nn.Linear(prev, n_out)
#             _init_orthogonal(last)
#             layers.append(last)
            
#             mlp = nn.Sequential(*layers)
#             mlp[:-1].apply(_init_orthogonal)
#             return mlp

#         self.phase_mlp_fourier = build_phase_mlp()
#         self.phase_mlp_pwc = build_phase_mlp()

#         # Learnable mixing angle theta
#         self.mix_theta = nn.Parameter(torch.tensor([math.pi / 4]))

#         # N-dimensional frequency matrix B
#         # Shape: (M, D). Sampled from Gaussian to allow isotropic high-frequency content.
#         B = torch.randn(n_modes, coords_dim, dtype=torch.float64) * freq_scale
#         self.register_buffer("_B", B, persistent=False)

#     def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
#         # t: (N,)   x: (N, D)
#         N, M, C = x.shape[0], self.n_modes, self.C

#         t_emb = self.time_embedding(t)

#         # --- Fourier Network Path ---
#         phases_f = self.phase_mlp_fourier(t_emb).reshape(N, M, C, 2)
#         cos_phi_f = phases_f[..., 0]                             
#         sin_phi_f = phases_f[..., 1]                             

#         # --- PWC Network Path ---
#         phases_p = self.phase_mlp_pwc(t_emb).reshape(N, M, C, 2)
#         cos_phi_p = phases_p[..., 0]                             
#         sin_phi_p = phases_p[..., 1]                             

#         # Spatial frequencies in N-D
#         # x is (N, D), _B is (M, D) -> angles is (N, M)
#         B = self._B.to(dtype=x.dtype)
#         angles = 2.0 * math.pi * (x @ B.T)
        
#         # Continuous Basis
#         S_f = torch.sin(angles)
#         C_f = torch.cos(angles)

#         # Soft Piecewise Constant Basis (Differentiable Square Waves)
#         # Using tanh(k * sin(x)) creates smooth, differentiable plateaus
#         S_p = torch.sign(self.pwc_steepness * S_f)
#         C_p = torch.sign(self.pwc_steepness * C_f)

#         # Calculate respective fields
#         v_fourier = (S_f[:, :, None] * cos_phi_f + C_f[:, :, None] * sin_phi_f).sum(dim=1)
#         v_fourier = v_fourier / math.sqrt(M)

#         v_pwc = (S_p[:, :, None] * cos_phi_p + C_p[:, :, None] * sin_phi_p).sum(dim=1)
#         v_pwc = v_pwc / math.sqrt(M)

#         # Apply learnable energy-preserving weights
#         w_fourier = torch.cos(self.mix_theta)
#         w_pwc = torch.sin(self.mix_theta)

#         return  (w_pwc * v_pwc) # (0.001 * v_fourier) +