import math
import warnings

import torch

from .base import InfiDictionary, orthogonal_synthesis


class FourierDictionary(InfiDictionary):
    """Infinite real Fourier dictionary with a finite learned synthesis block.

    Atoms are indexed by ``(k_1, ..., k_d, c)`` where each signed spatial
    frequency contributes a real 1-D factor:

    * ``k = 0``: constant ``1``.
    * ``k > 0``: ``sqrt(2) cos(2 pi k x)``.
    * ``k < 0``: ``sqrt(2) sin(2 pi |k| x)``.

    The spatial atom is the tensor product of those 1-D factors, placed in the
    selected output channel ``c``. The prior is the infinite isotropic power law
    ``P(k, c) proportional to (1 + ||k||_2^2)^(-steepness)``.

    A finite orthogonal synthesis matrix is learned over atoms whose PMF is at
    least ``synthesis_tail_probability``. Atoms outside that block remain the
    base Fourier atoms.
    """

    is_infinite = True

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        steepness: float = 2.0,
        m_max: int = 1024,
        synthesis_tail_probability: float = 1e-4,
        learn_synthesis: bool = True,
    ):
        super().__init__()
        if domain_dim < 1:
            raise ValueError(f"domain_dim must be >= 1; got {domain_dim}")
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1; got {num_channels}")
        if m_max < 0:
            raise ValueError(f"m_max must be non-negative; got {m_max}")
        if synthesis_tail_probability < 0.0:
            raise ValueError(
                "synthesis_tail_probability must be non-negative; "
                f"got {synthesis_tail_probability}"
            )
        self.domain_dim = int(domain_dim)
        self.num_channels = int(num_channels)
        self.steepness = float(steepness)
        self.m_max = int(m_max)
        self.synthesis_tail_probability = float(synthesis_tail_probability)
        self.learn_synthesis = bool(learn_synthesis)

        if self.steepness <= 0.5 * self.domain_dim:
            warnings.warn(
                f"steepness={steepness} <= d/2 = {0.5 * self.domain_dim}; the "
                "power-law normalizer would diverge without the finite m_max "
                "approximation.",
                RuntimeWarning,
                stacklevel=2,
            )

        l2_w = self._precompute_l2_shell_weights()
        self._Z_L2 = l2_w.sum().item()
        self._l2_shell_cdf = (l2_w / self._Z_L2).cumsum(0)

        self.register_buffer(
            "_synthesis_indices",
            self.get_high_probability_indices(self.synthesis_tail_probability),
            persistent=False,
        )
        self._synthesis = orthogonal_synthesis(self._synthesis_indices.shape[0], self.learn_synthesis)

    # -- Shell helpers -----------------------------------------------------

    def _precompute_l2_shell_weights(self) -> torch.Tensor:
        """Shell weights for the approximate power-law normalizer."""
        d = self.domain_dim
        alpha = self.steepness
        W = torch.zeros(self.m_max + 1)
        W[0] = 1.0

        for m in range(1, self.m_max + 1):
            if d == 1:
                W[m] = 2.0 * float((1.0 + m * m) ** (-alpha))
            elif d == 2:
                k2 = torch.arange(-m, m + 1).float()
                w_rows = 2.0 * (1.0 + m * m + k2 * k2).pow(-alpha).sum().item()
                k1 = torch.arange(-(m - 1), m).float()
                w_cols = 2.0 * (1.0 + k1 * k1 + m * m).pow(-alpha).sum().item()
                W[m] = w_rows + w_cols
            else:
                shell_sz = float((2 * m + 1) ** d - (2 * m - 1) ** d)
                W[m] = shell_sz * float((1.0 + m * m) ** (-alpha))

        return W

    def _sample_from_shells_l2(self, shells: torch.Tensor) -> torch.Tensor:
        """Sample spatial indices from L-infinity shells with L2 power weights."""
        S = shells.shape[0]
        d = self.domain_dim
        result = torch.zeros(S, d, dtype=torch.long, device=shells.device)
        done = torch.zeros(S, dtype=torch.bool, device=shells.device)

        while not done.all():
            m = shells
            cands = torch.zeros(S, d, dtype=torch.long, device=shells.device)
            for dim in range(d):
                r = torch.rand(S, device=shells.device)
                cands[:, dim] = (r * (2 * m.float() + 1)).long() - m

            on_shell = cands.abs().amax(dim=-1) == m
            l2sq = cands.pow(2).sum(dim=-1).float()
            m2 = m.float().pow(2)
            ratio = torch.where(
                m == 0,
                torch.ones(S, device=shells.device),
                ((1.0 + m2) / (1.0 + l2sq)).pow(self.steepness),
            )
            accept = on_shell & (torch.rand(S, device=shells.device) < ratio) & ~done
            result[accept] = cands[accept]
            done = done | accept

        return result.cpu()

    # -- Coordinate and atom helpers --------------------------------------

    def _get_base_spatial_atoms(
        self,
        coords: torch.Tensor,
        spatial_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate tensor-product real Fourier spatial atoms."""
        spatial_idx = spatial_idx.to(coords.device)
        vals = torch.ones(
            (spatial_idx.shape[0], coords.shape[0]),
            device=coords.device,
            dtype=coords.dtype,
        )

        scale = math.sqrt(2.0)
        for dim in range(self.domain_dim):
            freq = spatial_idx[:, dim]
            x = coords[:, dim][None, :]
            factor = torch.ones_like(vals)

            pos = freq > 0
            if pos.any():
                k = freq[pos].to(coords.dtype)[:, None]
                factor[pos] = scale * torch.cos(2.0 * math.pi * k * x)

            neg = freq < 0
            if neg.any():
                k = (-freq[neg]).to(coords.dtype)[:, None]
                factor[neg] = scale * torch.sin(2.0 * math.pi * k * x)

            vals = vals * factor

        return vals

    def _get_base_atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        spatial_idx = idx[:, :-1]
        channel_idx = idx[:, -1].long()
        A, N = idx.shape[0], coords.shape[0]
        C = self.num_channels

        phi = self._get_base_spatial_atoms(coords, spatial_idx)
        vals = torch.zeros((A, N, C), device=coords.device, dtype=phi.dtype)
        valid = (channel_idx >= 0) & (channel_idx < C)
        if valid.any():
            rows = valid.nonzero(as_tuple=False).squeeze(-1)
            vals[rows, :, channel_idx[valid]] = phi[rows]
        return vals

    def _synthesis_positions(self, idx: torch.Tensor) -> torch.Tensor:
        if self._synthesis_indices.numel() == 0:
            return torch.full((idx.shape[0],), -1, device=idx.device, dtype=torch.long)

        block = self._synthesis_indices.to(idx.device)
        matches = (idx[:, None, :] == block[None, :, :]).all(dim=-1)
        found = matches.any(dim=-1)
        pos = matches.to(torch.long).argmax(dim=-1)
        return torch.where(found, pos, torch.full_like(pos, -1))

    # -- Core dictionary methods ------------------------------------------

    def sample_indices(
        self,
        num_samples: int,
        with_replacement: bool = True,
    ) -> torch.Tensor:
        """Sample indices from the infinite power-law prior."""
        if not with_replacement:
            raise NotImplementedError(
                "without-replacement sampling is undefined for the infinite "
                "power-law prior"
            )

        u = torch.rand(num_samples)
        shells = torch.searchsorted(self._l2_shell_cdf, u).clamp(0, self.m_max)
        spatial = self._sample_from_shells_l2(shells)
        channels = torch.randint(0, self.num_channels, (num_samples,))
        return torch.cat([spatial, channels.unsqueeze(-1)], dim=-1)

    def get_atoms(
        self,
        coords: torch.Tensor,
        idx: torch.Tensor,
        synthesis: bool = True,
    ) -> torch.Tensor:
        """Evaluate real Fourier atoms, optionally applying the synthesis block."""
        device = coords.device
        idx = idx.to(device)
        vals = self._get_base_atoms(coords, idx)

        if not synthesis or self._synthesis is None or idx.shape[0] == 0:
            return vals

        pos = self._synthesis_positions(idx)
        in_block = pos >= 0
        if not in_block.any():
            return vals

        block_atoms = self._get_base_atoms(coords, self._synthesis_indices.to(device))
        weight = self._synthesis.weight.to(dtype=coords.dtype, device=device)
        mixed = weight[pos[in_block]] @ block_atoms.reshape(block_atoms.shape[0], -1)
        vals[in_block] = mixed.reshape(in_block.sum().item(), coords.shape[0], self.num_channels)
        return vals

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        spatial_idx = idx[:, :-1]
        channel_idx = idx[:, -1]
        valid = (channel_idx >= 0) & (channel_idx < self.num_channels)
        l2sq = spatial_idx.pow(2).sum(dim=-1).float()
        base = (1.0 + l2sq).pow(-self.steepness) / self._Z_L2 / self.num_channels
        out = torch.zeros(idx.shape[0], dtype=base.dtype, device=idx.device)
        out[valid] = base[valid]
        return out

    def _compute_M_bound(self, tail_probability: float) -> int:
        """Smallest box half-width containing every atom with PMF >= the threshold.

        The most probable atom on the L-infinity shell ``max|k_i| = m`` sits on
        an axis, at ``||k||^2 = m^2``; once even that one falls below the
        threshold, so does every atom on this and all outer shells.
        """
        scale = 1.0 / self.num_channels / self._Z_L2
        last_m = 0
        for m in range(self.m_max + 1):
            min_l2sq = float(m) ** 2
            atom_pmf = float((1.0 + min_l2sq) ** (-self.steepness)) * scale
            if atom_pmf < tail_probability:
                return last_m
            last_m = m
        return self.m_max

    def _box_indices(self, max_freq: int) -> torch.Tensor:
        """All ``(k_1, ..., k_d, c)`` with ``|k_i| <= max_freq`` for every axis."""
        vals = torch.arange(-max_freq, max_freq + 1)
        grids = torch.meshgrid(*[vals] * self.domain_dim, indexing="ij")
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)

        C = self.num_channels
        A_s = spatial_idx.shape[0]
        channels = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = (
            spatial_idx.unsqueeze(1)
            .expand(-1, C, -1)
            .reshape(-1, self.domain_dim)
        )
        return torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all indices with prior PMF at least ``tail_probability``.

        The selected set is an isotropic ``||k||`` ball in frequency space (the
        PMF depends only on ``||k||``), hence rotation-closed.
        """
        M = self._compute_M_bound(float(tail_probability))
        idx = self._box_indices(M)
        return idx[self.get_index_pmfs(idx) >= tail_probability]

    def get_lowpass_indices(self, max_freq: int) -> torch.Tensor:
        """Return all atoms inside the axis-aligned low-pass box ``|k_i| <= max_freq``.

        This is a separable per-axis frequency cutoff (an L-infinity box in
        frequency), in contrast to the isotropic ``||k||`` ball of
        :meth:`get_high_probability_indices`. The box is *not* rotation-closed,
        so captured energy and reconstructions under it are sensitive to
        rotation of the input -- useful for probing rotation equivariance.
        """
        max_freq = int(max_freq)
        if max_freq < 0:
            raise ValueError(f"max_freq must be non-negative; got {max_freq}")
        return self._box_indices(max_freq)
