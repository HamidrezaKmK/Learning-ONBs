import itertools
import math
import torch
from torch import nn

from .base import NeuralField, _init_orthogonal, RMSNorm
from .time_embedding import SinusoidalTimeEmbedding


class FiLMedMLP(nn.Module):
    """Feedforward MLP with Feature-wise Linear Modulation (FiLM) at every hidden layer.

    At each hidden layer the conditioning vector ``cond`` is projected to a
    per-feature scale ``γ`` and shift ``β``, applied after RMSNorm::

        h = activation( (1 + γ(cond)) · RMSNorm(linear(x)) + β(cond) )

    The ``(1 + γ)`` residual parameterisation means zero-initialised projection
    weights leave the network as a plain MLP at startup; conditioning is
    gradually learned on top of that baseline.  The output layer applies
    ``BatchNorm1d(affine=False)`` for zero-mean unit-variance output.

    Args:
        in_dim:      Input feature dimension.
        out_dim:     Output feature dimension.
        hidden_dims: Width of each hidden layer.
        cond_dim:    Dimension of the conditioning vector.
        activation:  Activation class applied after each FiLM step.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: tuple,
        cond_dim: int,
        activation,
    ):
        super().__init__()
        self.linears    = nn.ModuleList()
        self.norms      = nn.ModuleList()
        self.film_projs = nn.ModuleList()
        self.acts       = nn.ModuleList()

        prev = in_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            self.norms.append(RMSNorm())
            proj = nn.Linear(cond_dim, 2 * h)
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)
            self.film_projs.append(proj)
            self.acts.append(activation())
            prev = h

        self.out_linear = nn.Linear(prev, out_dim)
        self.out_bn     = nn.BatchNorm1d(out_dim, affine=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x:    (N, in_dim)
        # cond: (N, cond_dim)
        for linear, norm, film_proj, act in zip(
            self.linears, self.norms, self.film_projs, self.acts
        ):
            h = norm(linear(x))
            gamma, beta = film_proj(cond).chunk(2, dim=-1)
            x = act((1.0 + gamma) * h + beta)
        return self.out_bn(self.out_linear(x))

class FourierDistortedField(NeuralField):
    """A time-conditioned neural field that adds a random-Fourier distortion to an inner field.

    Wraps an ``inner_field`` and adds a structured spectral distortion whose
    frequency sweeps from ``low_freq`` to ``high_freq`` as the ODE integrates
    from ``low_t`` to ``high_t``.  The distortion is mixed with the RMS-normalised
    inner field output at a fixed 45° angle::

        feat_i(t, x) = sin(freq(t) · w_i^T x + phase_i(t))
        v_distortion  = Σ_i  coeff_i(t, c) · feat_i(t, x)        shape (N, C)
        out = cos(π/4) · (v_inner / RMS(v_inner)) + sin(π/4) · v_distortion

    **Modes.**  The actual number of modes is ``n_modes^d`` (a tensor-product
    grid) rather than ``n_modes``.

    **Direction vectors.**  Each ``w_i`` is a random Gaussian vector perturbed
    to introduce both directional diversity and magnitude variation::

        w_i = w_raw / ‖w_raw‖ + ε,   ε ~ N(0, I)

    These are fixed non-trainable buffers.

    **Frequency schedule.**  The shared scalar frequency follows a
    ``gamma``-shaped curve::

        t_scaled  = (clamp(t, low_t, high_t) - low_t) / (high_t - low_t)
        freq(t)   = low_freq + (1 - (1 - t_scaled)^gamma) · (high_freq - low_freq)

    Higher ``gamma`` → frequency ramps up slowly then accelerates;
    ``gamma = 1`` → linear sweep.

    **Coefficient normalisation.**  ``coeff_i(t, c)`` are unit-normalised over
    the modes axis per output channel before the weighted sum, so the distortion
    energy is independent of the MLP output scale.

    **Learnable flag.**  When ``learnable=False`` both ``time_to_phases`` and
    ``time_to_coeffs`` are frozen at random initialisation.  Only ``inner_field``
    is optimised, and it must learn to compensate for the fixed distortion.

    Args:
        coords_dim:          Spatial dimension ``d``.
        output_dim:          Number of output channels ``C``.
        inner_field_partial: Callable ``(coords_dim, output_dim) → NeuralField``
                             that constructs the wrapped inner field.
        n_modes:             Base number of modes; actual count is ``n_modes^d``.
        learnable:           If ``False``, phase and coefficient MLPs are frozen.
        time_hidden_dims:    Hidden layer widths for both time-conditioned MLPs.
        n_time_freqs:        Number of sinusoidal frequencies in the time embedding.
        low_freq:            Spatial frequency at the start of the sweep (``t = low_t``).
        high_freq:           Spatial frequency at the end of the sweep (``t = high_t``).
        low_t:               ODE time value corresponding to ``freq = low_freq``.
        high_t:              ODE time value corresponding to ``freq = high_freq``.
        gamma:               Exponent controlling the frequency ramp shape (``γ > 0``).
        activation:          Activation class used between hidden layers of both MLPs.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        inner_field_partial: callable,
        n_modes: int,
        learnable: bool = True,
        time_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        low_freq: float = 1.0,
        high_freq: float = 32.0,
        low_t: float = 0.0,
        high_t: float = 1.0,
        gamma: float = 1.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.inner_field: NeuralField = inner_field_partial(coords_dim=coords_dim, output_dim=output_dim)
        self.n_modes   = n_modes ** coords_dim
        self.C         = output_dim
        self.d         = coords_dim
        self.low_freq  = low_freq
        self.high_freq = high_freq
        self.low_t     = low_t
        self.high_t    = high_t
        self.learnable = learnable
        self.gamma     = gamma 
        
        # Fixed unit-norm random direction vectors w_i ~ Gaussian with d dimensions.
        w = torch.randn(self.n_modes, coords_dim) 
        w = w / torch.norm(w, dim=-1, keepdim=True).clamp(1e-8) + torch.randn_like(w)
        self.register_buffer("w", w)

        # Time embedding has no learnable parameters (only a frequency buffer).
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        # MLP: t_emb → self.n_modes phases
        self.time_to_phases = self._build_mlp(emb_dim, self.n_modes, time_hidden_dims, activation)
        # MLP: t_emb → self.n_modes * C combination coefficients (one per mode-channel pair)
        self.time_to_coeffs = self._build_mlp(emb_dim, self.n_modes * output_dim, time_hidden_dims, activation)

        if learnable:
            self.time_to_phases.apply(_init_orthogonal)
            self.time_to_coeffs.apply(_init_orthogonal)
        else:
            # Freeze both MLPs — distortion is a fixed random function of t.
            for p in itertools.chain(self.time_to_phases.parameters(),
                                     self.time_to_coeffs.parameters()):
                p.requires_grad_(False)

    @staticmethod
    def _build_mlp(in_dim: int, out_dim: int, hidden_dims: tuple, activation) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        return nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        # Per-point time embedding: (N, emb_dim)
        t_emb = self.time_embedding(t)

        # Per-point normalised time and centre frequency: both (N, 1)
        t_scaled = ((t.clamp(self.low_t, self.high_t) - self.low_t)
                    / (self.high_t - self.low_t)).unsqueeze(-1)
        freq_scale = 1 - (1 - t_scaled) ** self.gamma
        freq = self.low_freq + freq_scale * (self.high_freq - self.low_freq)  # (N, 1)

        # Random projections w_i^T x_n: (N, n_modes)
        proj = x @ self.w.T

        if self.learnable:
            phases = self.time_to_phases(t_emb)                                # (N, n_modes)
            coeffs = self.time_to_coeffs(t_emb).view(N, self.n_modes, self.C) # (N, n_modes, C)
        else:
            with torch.no_grad():
                phases = self.time_to_phases(t_emb)
                coeffs = self.time_to_coeffs(t_emb).view(N, self.n_modes, self.C)

        # Fourier features: sin(freq * w_i^T x + phase_i(t)) → (N, n_modes)
        features = torch.sin(freq * proj + phases)

        # Normalise coefficients over the modes axis so each output channel has
        # unit-norm weights regardless of the MLP output scale.
        coeffs = coeffs / coeffs.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # Weighted sum over modes → (N, C)
        v_distortion = (features.unsqueeze(-1) * coeffs).sum(dim=1)


        v_inner = self.inner_field(t, x)
        v_inner_rms      = v_inner.pow(2).mean().sqrt().clamp(min=1e-8)
        v_inner_rescaled = v_inner / v_inner_rms

        # return v_distortion
        return math.cos(math.pi / 4) * v_inner_rescaled + math.sin(math.pi / 4) * v_distortion


class FourierDistortedFieldV2(NeuralField):
    """A frequency-sweeping Fourier neural field with a coarse-to-fine inductive bias.

    The core design principle is that as the ODE time ``t`` progresses from
    ``low_t`` to ``high_t``, the network's spatial frequency ``freq(t)`` sweeps
    from ``low_freq`` to ``high_freq``.  This encodes the inductive bias that
    early in the flow the field should capture coarse, low-frequency structure,
    while later timesteps progressively refine high-frequency detail — mirroring
    the coarse-to-fine nature of many natural signals.

    At each point ``(t_n, x_n)`` it computes ``n_modes^d`` random Fourier
    features whose spatial directions and phases are both time-dependent, then
    feeds them (together with ``x``) through a spatial MLP to produce the
    output::

        W_i(t)    = time_to_freqs(t_emb)[i]  ∈ R^d,  normalised to unit norm
        φ_i(t)    = time_to_phases(t_emb)[i] ∈ R
        proj_i    = W_i(t)^T x
        feat_i    = [sin(freq(t) · proj_i + φ_i),
                     cos(freq(t) · proj_i + φ_i)]
        v         = ff(cat([x, {feat_i}_i]))              shape (N, C)

    **Modes.**  The actual number of modes is ``n_modes^d`` (tensor-product
    grid); each mode produces one sin and one cos feature, so the ``ff`` MLP
    receives ``d + 2 · n_modes^d`` inputs.

    **Time-dependent directions.**  ``W_i(t)`` is the output of ``time_to_freqs``
    normalised to the unit sphere, so the projection ``proj_i = W_i(t)^T x`` is
    always in ``[-‖x‖, ‖x‖]`` regardless of the MLP output scale.

    **Frequency schedule.**  The shape of the coarse-to-fine ramp is controlled
    by ``gamma``::

        t_scaled = (clamp(t, low_t, high_t) - low_t) / (high_t - low_t)
        freq(t)  = low_freq + freq_scale · (high_freq - low_freq)

        freq_scale = t_scaled^gamma               if gamma ≥ 0   (slow start, accelerating)
        freq_scale = 1 - (1 - t_scaled)^(-gamma)  if gamma < 0   (fast start, decelerating)

    **Spatial MLP.**  ``ff`` maps ``[x, features]`` to the output with
    ``RMSNorm`` between hidden layers and a ``BatchNorm1d(affine=False)`` at the
    output, ensuring zero-mean unit-variance output across the spatial batch.

    Args:
        coords_dim:          Spatial dimension ``d``.
        output_dim:          Number of output channels ``C``.
        n_modes:             Base number of modes; actual count is ``n_modes^d``.
        spatial_hidden_dims: Hidden layer widths for the spatial MLP (``ff``).
        time_hidden_dims:    Hidden layer widths for the two time-conditioned MLPs.
        n_time_freqs:        Number of sinusoidal frequencies in the time embedding.
        low_freq:            Spatial frequency at the start of the sweep (``t = low_t``).
        high_freq:           Spatial frequency at the end of the sweep (``t = high_t``).
        low_t:               ODE time value corresponding to ``freq = low_freq``.
        high_t:              ODE time value corresponding to ``freq = high_freq``.
        gamma:               Controls the ramp shape. Positive → slow start (more
                             time spent at low frequencies); negative → fast start
                             (quickly reaches high frequencies).
        activation:          Activation class used between hidden layers.
    """

    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_modes: int,
        freq_strides: int = 1,
        spatial_hidden_dims: tuple = (64, 64),
        time_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        low_freq: float = 1.0,
        high_freq: float = 32.0,
        low_t: float = 0.0,
        high_t: float = 1.0,
        gamma: float = 1.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.n_modes   = n_modes ** coords_dim
        self.C         = output_dim
        self.d         = coords_dim
        self.low_freq  = low_freq
        self.high_freq = high_freq
        self.low_t     = low_t
        self.high_t    = high_t
        self.gamma     = gamma 

        if self.n_modes < self.d:
            raise ValueError("n_modes must be at least as large as coords_dim for the QR orthogonalisation to work.")
        
        # Time embedding has no learnable parameters (only a frequency buffer).
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim
        self.freq_strides = freq_strides
        self.K = 2 * self.freq_strides  # Number of fourier feature frequencies per timestep

        # MLP: t_emb → self.n_modes phases
        self.time_to_phases = self._build_mlp(
            emb_dim,
            self.n_modes * self.K, 
            time_hidden_dims, 
            activation, 
            use_batchnorm=False, 
            use_rmsnorm=True, 
            bias=True,
        )
        # MLP: t_emb → self.n_modes * d combination coefficients (one per mode-coordinate pair)
        self.time_to_freqs = self._build_mlp(
            emb_dim, 
            self.n_modes * coords_dim * self.K, 
            time_hidden_dims, 
            activation, 
            use_batchnorm=False, 
            use_rmsnorm=True, 
            bias=False,
        )
        # Spatial MLP with FiLM conditioning from the time embedding.
        self.ff = self._build_mlp(
            emb_dim + coords_dim + self.n_modes * 2 * self.K, 
            output_dim, 
            spatial_hidden_dims, 
            activation, 
            use_batchnorm=True, 
            use_rmsnorm=True, 
            bias=True,
        )

        self.time_to_phases.apply(_init_orthogonal)
        self.time_to_freqs.apply(_init_orthogonal)

        # _init_orthogonal zeros all biases, so time_to_phases(t_emb) starts as
        # a purely linear projection of t_emb: all modes' phases are correlated
        # linear combinations of the same structured vector.  Re-randomising the
        # last layer's bias spreads the initial outputs across [0, 2π] so each
        # mode starts at an independent random phase.
        nn.init.uniform_(self.time_to_phases[-1].bias, 0, 1)


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

    def _compute_freq(self, t: torch.Tensor) -> torch.Tensor:
        # Per-point normalised time and centre frequency: both (N, 1)
        t_scaled = ((t.clamp(self.low_t, self.high_t) - self.low_t)
                    / (self.high_t - self.low_t)).unsqueeze(-1)
        t_scaled = t_scaled.repeat(1, self.freq_strides)  # (N, K/2)
        t_scaled = t_scaled + torch.arange(self.freq_strides, device=t.device).unsqueeze(0) / self.freq_strides  # (N, K/2), add strides
        t_scaled = t_scaled % 1.0  # Wrap around to [0, 1]
        t_scaled = torch.cat([t_scaled, 1 - t_scaled], dim=-1)  # (N, K)
        if self.gamma < 0:
            freq_scale = 1 - (1 - t_scaled) ** (-self.gamma)
        else:
            freq_scale = t_scaled ** self.gamma
        freq = self.low_freq + freq_scale * (self.high_freq - self.low_freq)  # (N, K)
        return freq
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        # Per-point time embedding: (N, emb_dim)
        t_emb = self.time_embedding(t)

        phases = self.time_to_phases(t_emb).view(N, self.n_modes, self.K)   # (N, n_modes, K)
        W = self.time_to_freqs(t_emb).view(N, self.n_modes, self.K, self.d) # (N, n_modes, K, d)
        W_orth = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # Normalize to unit vectors
        
        freq = self._compute_freq(t).view(N, 1, self.K)  # (N, 1, K)
        proj = torch.einsum('nmkd,nd->nmk', W_orth, x)  # (N, n_modes, K)

        arg = 2 * math.pi * freq * (proj + phases)  # (N, n_modes, K)
        sin_features = torch.sin(arg.view(N, self.n_modes * self.K)) # frequency adjusted features
        cos_features = torch.cos(arg.view(N, self.n_modes * self.K)) # frequency adjusted features


        # Concatenate Fourier features with input coordinates and pass through FiLM-conditioned MLP.
        ff_input = torch.cat([t_emb, x, sin_features, cos_features], dim=-1)  # (N, emb_dim + d + 4 * n_modes)
        v = self.ff(ff_input)  # (N, C)
        
        return v

