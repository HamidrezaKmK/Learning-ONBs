import itertools
import math
import torch
from torch import nn

from .base import TimeEvolvingField, _init_orthogonal, RMSNorm
from .time_embedding import SinusoidalTimeEmbedding
from infidictionary.dictionaries import FourierDictionary

class FourierDistortedFieldV2(TimeEvolvingField):
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
        # the actual networks that should match the data
        self.ff1 = self._build_mlp(
            emb_dim + coords_dim + self.n_modes * self.K, 
            output_dim, 
            spatial_hidden_dims, 
            activation, 
            use_batchnorm=True, 
            use_rmsnorm=True, 
            bias=True,
        )
        self.ff2 = self._build_mlp(
            emb_dim + coords_dim + self.n_modes * self.K, 
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
                    / (self.high_t - self.low_t + 1e-3)).unsqueeze(-1)
        t_scaled = t_scaled + torch.arange(self.freq_strides, device=t.device).unsqueeze(0) / self.freq_strides  # (N, K/2), add strides
        t_scaled = t_scaled % 1.0  # Wrap around to [0, 1]
        t_scaled = torch.cat([t_scaled, 1 - t_scaled], dim=-1)  # (N, K)
        if self.gamma < 0:
            freq_scale = 1 - (1 - t_scaled) ** (-self.gamma)
        else:
            freq_scale = t_scaled ** self.gamma
        freq = self.low_freq + freq_scale * (self.high_freq - self.low_freq)  # (N, K)
        return freq[:, :self.K//2], freq[:, self.K//2:]  # Return both halves of the frequencies for potential use in sin and cos features
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        # Per-point time embedding: (N, emb_dim)
        t_emb = self.time_embedding(t)

        phases = self.time_to_phases(t_emb).view(N, self.n_modes, self.K)   # (N, n_modes, K)
        W = self.time_to_freqs(t_emb).view(N, self.n_modes, self.K, self.d) # (N, n_modes, K, d) 
        W_orth = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # Normalize to unit vectors

        # pass W through a softmax        
        freq_ff1, freq_ff2 = self._compute_freq(t)  # (N, K/2), (N, K/2)
        freq_ff1 = freq_ff1.unsqueeze(1)  # (N, 1, K/2)
        freq_ff2 = freq_ff2.unsqueeze(1)  # (N, 1, K/2)
        proj = torch.einsum('nmkd,nd->nmk', W_orth, x)  # (N, n_modes, K)
        proj1, proj2 = proj[:, :, :self.K//2], proj[:, :, self.K//2:]  # (N, n_modes, K/2), (N, n_modes, K/2)
        phases1, phases2 = phases[:, :, :self.K//2], phases[:, :, self.K//2:]  # (N, n_modes, K/2), (N, n_modes, K/2)

        arg1 = 2 * math.pi * freq_ff1 * (proj1 + phases1)  # (N, n_modes, K/2)
        fourier_feature1 = torch.cat([torch.sin(arg1), torch.cos(arg1)], dim=-1)  # (N, n_modes, K)
        fourier_feature1 = fourier_feature1.view(N, self.n_modes * self.K)  # (N, n_modes * K)
        field1 = self.ff1(torch.cat([t_emb, x, fourier_feature1], dim=-1))  # (N, C)

        arg2 = 2 * math.pi * freq_ff2 * (proj2 + phases2)  # (N, n_modes, K/2)
        fourier_feature2 = torch.cat([torch.sin(arg2), torch.cos(arg2)], dim=-1)  # (N, n_modes, K)
        fourier_feature2 = fourier_feature2.view(N, self.n_modes * self.K)  # (N, n_modes * K)
        field2 = self.ff2(torch.cat([t_emb, x, fourier_feature2], dim=-1))  # (N, C)

        return field1, field2



class FourierDistortedFieldV3(TimeEvolvingField):
    def __init__(
        self,
        coords_dim: int,
        output_dim: int,
        n_truncation: int,
        spatial_hidden_dims: tuple = (64, 64),
        n_time_freqs: int = 8,
        low_t: float = 0.0,
        high_t: float = 1.0,
        activation=nn.SiLU,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)

        self.C         = output_dim
        self.d         = coords_dim
        self.low_t     = low_t
        self.high_t    = high_t
        self.fourier_dict = FourierDictionary(
            domain_dim=coords_dim,
            num_channels=output_dim,
            p=0.3,
        )
        indices_of_interest = self.fourier_dict.get_truncated_indices(n_truncation)
        # shuffle indices
        self.register_buffer("indices_alpha", indices_of_interest[torch.randperm(len(indices_of_interest))])
        self.register_buffer("indices_beta", indices_of_interest[torch.randperm(len(indices_of_interest))])

        # Time embedding has no learnable parameters (only a frequency buffer).
        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        self.ff1 = self._build_mlp(
            emb_dim + coords_dim,
            output_dim, 
            spatial_hidden_dims, 
            activation, 
            use_batchnorm=True, 
            use_rmsnorm=True, 
            bias=True,
        )
        self.ff2 = self._build_mlp(
            emb_dim + coords_dim,
            output_dim,
            spatial_hidden_dims,
            activation,
            use_batchnorm=True,
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

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (N,)   x: (N, d)
        N = x.shape[0]

        t_emb = self.time_embedding(t)
        
        # convert time to index
        with torch.no_grad():
            alpha_vals = self.fourier_dict.get_atoms(x.detach(), self.indices_alpha).detach()  # (num_indices, N, C)
            beta_vals = self.fourier_dict.get_atoms(x.detach(), self.indices_beta).detach()    # (num_indices, N, C)
            range_N = torch.arange(N, device=x.device).detach()

            t_scaled = ((t.clamp(self.low_t, self.high_t) - self.low_t) / (self.high_t - self.low_t + 1e-3))
            t_idx_alpha = torch.floor(t_scaled * len(self.indices_alpha)).long().clamp(max=len(self.indices_alpha)-1)  # (N,)
            t_idx_beta = torch.floor(t_scaled * len(self.indices_beta)).long().clamp(max=len(self.indices_beta)-1)  # (N,)
            alpha_base = alpha_vals[t_idx_alpha, range_N]  # (N, C)
            beta_base = beta_vals[t_idx_beta, range_N]    # (N, C)
        

        # Per-point time embedding: (N, emb_dim)
        alpha_learnable = self.ff1(torch.cat([t_emb, x], dim=-1))  # (N, C)
        beta_learnable = self.ff2(torch.cat([t_emb, x], dim=-1))  # (N, C)

        v1 = math.sin(math.pi / 4) * (alpha_base + alpha_learnable)
        v2 = math.sin(math.pi / 4) * (beta_base + beta_learnable)


        return v1, v2

