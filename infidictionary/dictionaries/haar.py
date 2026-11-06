"""Isotropic tensor-product Haar wavelet dictionary on ``[0, 1]^d``."""

import math
import torch

from .base import InfiDictionary, orthogonal_synthesis


class HaarWaveletDictionary(InfiDictionary):
    """Infinite Haar basis indexed by ``[level, shifts..., orientation, channel]``.

    ``level=-1`` denotes the per-channel constant scaling function.  At level
    ``j >= 0``, ``shifts`` lies in ``[0, 2**j)^d`` and ``orientation`` is a
    non-zero bit mask: bit ``i`` selects the 1-D Haar wavelet on axis ``i``.
    The PMF gives total mass ``scaling_mass`` to scaling atoms and distributes
    the remaining mass geometrically over wavelet levels.
    """

    is_infinite = True

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        scaling_mass: float = 0.5,
        level_decay: float = 0.5,
        synthesis_tail_probability: float = 1e-4,
        learn_synthesis: bool = True,
    ):
        super().__init__()
        if domain_dim < 1 or num_channels < 1:
            raise ValueError("domain_dim and num_channels must both be >= 1")
        if not 0.0 < scaling_mass < 1.0:
            raise ValueError("scaling_mass must be in (0, 1)")
        if not 0.0 < level_decay < 1.0:
            raise ValueError("level_decay must be in (0, 1)")
        if synthesis_tail_probability <= 0.0:
            raise ValueError("synthesis_tail_probability must be positive for the infinite Haar prior")
        self.domain_dim = int(domain_dim)
        self.num_channels = int(num_channels)
        self.scaling_mass = float(scaling_mass)
        self.level_decay = float(level_decay)
        self.synthesis_tail_probability = float(synthesis_tail_probability)
        self.learn_synthesis = bool(learn_synthesis)
        self.register_buffer(
            "_synthesis_indices",
            self.get_high_probability_indices(synthesis_tail_probability),
            persistent=False,
        )
        self._synthesis = orthogonal_synthesis(self._synthesis_indices.shape[0], self.learn_synthesis)

    @property
    def _index_width(self) -> int:
        return self.domain_dim + 3

    def _wavelet_pmf(self, level: torch.Tensor) -> torch.Tensor:
        count = self.num_channels * (2 ** self.domain_dim - 1) * (2 ** (self.domain_dim * level))
        return ((1.0 - self.scaling_mass) * (1.0 - self.level_decay) * self.level_decay ** level) / count

    def _base_atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        A, N = idx.shape[0], coords.shape[0]
        values = torch.zeros(A, N, self.num_channels, dtype=coords.dtype, device=coords.device)
        if idx.ndim != 2 or idx.shape[1] != self._index_width:
            raise ValueError(f"idx must have shape (A, {self._index_width})")
        level, shifts = idx[:, 0].long(), idx[:, 1:1 + self.domain_dim].long()
        orientation, channel = idx[:, -2].long(), idx[:, -1].long()
        valid_channel = (channel >= 0) & (channel < self.num_channels)
        scaling = (level == -1) & (shifts == 0).all(dim=1) & (orientation == 0) & valid_channel
        if scaling.any():
            rows = scaling.nonzero(as_tuple=False).squeeze(-1)
            values[rows, :, channel[rows]] = 1.0
        for row in ((level >= 0) & valid_channel).nonzero(as_tuple=False).flatten().tolist():
            j = int(level[row])
            cells = 2 ** j
            orient = int(orientation[row])
            if orient < 1 or orient >= 2 ** self.domain_dim or (shifts[row] < 0).any() or (shifts[row] >= cells).any():
                continue
            local = coords * cells - shifts[row].to(coords.dtype)
            factor = torch.ones(N, dtype=coords.dtype, device=coords.device)
            for axis in range(self.domain_dim):
                inside = (local[:, axis] >= 0) & (local[:, axis] < 1)
                if (orient >> axis) & 1:
                    axis_value = torch.where(local[:, axis] < 0.5, 1.0, -1.0)
                else:
                    axis_value = torch.ones(N, dtype=coords.dtype, device=coords.device)
                factor = factor * axis_value * inside
            values[row, :, channel[row]] = factor * (cells ** (self.domain_dim / 2))
        return values

    def get_atoms(self, coords: torch.Tensor, idx: torch.Tensor, synthesis: bool = True) -> torch.Tensor:
        idx = idx.to(coords.device)
        values = self._base_atoms(coords, idx)
        if not synthesis or self._synthesis is None or idx.shape[0] == 0:
            return values
        block_idx = self._synthesis_indices.to(coords.device)
        matches = (idx[:, None, :] == block_idx[None, :, :]).all(dim=-1)
        inside = matches.any(dim=1)
        if inside.any():
            positions = matches.long().argmax(dim=1)
            block = self._base_atoms(coords, block_idx)
            weight = self._synthesis.weight.to(dtype=coords.dtype, device=coords.device)
            mixed = weight[positions[inside]] @ block.reshape(block.shape[0], -1)
            values[inside] = mixed.reshape(inside.sum().item(), coords.shape[0], self.num_channels)
        return values

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        idx = idx.to(torch.long)
        out = torch.zeros(idx.shape[0], dtype=torch.get_default_dtype(), device=idx.device)
        if idx.ndim != 2 or idx.shape[1] != self._index_width:
            raise ValueError(f"idx must have shape (A, {self._index_width})")
        level, shifts = idx[:, 0], idx[:, 1:1 + self.domain_dim]
        orientation, channel = idx[:, -2], idx[:, -1]
        valid_channel = (channel >= 0) & (channel < self.num_channels)
        scaling = (level == -1) & (shifts == 0).all(dim=1) & (orientation == 0) & valid_channel
        out[scaling] = self.scaling_mass / self.num_channels
        wavelet = (level >= 0) & (orientation >= 1) & (orientation < 2 ** self.domain_dim) & valid_channel
        if wavelet.any():
            rows = wavelet.nonzero(as_tuple=False).squeeze(-1)
            cells = 2 ** level[rows, None]
            valid_shift = (shifts[rows] >= 0).all(dim=1) & (shifts[rows] < cells).all(dim=1)
            rows = rows[valid_shift]
            out[rows] = self._wavelet_pmf(level[rows])
        return out

    def _level_indices(self, level: int) -> torch.Tensor:
        cells = 2 ** level
        axes = torch.arange(cells)
        shifts = torch.stack(torch.meshgrid(*([axes] * self.domain_dim), indexing="ij"), dim=-1).reshape(-1, self.domain_dim)
        orientations = torch.arange(1, 2 ** self.domain_dim)
        channels = torch.arange(self.num_channels)
        base = shifts[:, None, None, :].expand(-1, len(orientations), self.num_channels, -1).reshape(-1, self.domain_dim)
        orient = orientations[None, :, None].expand(shifts.shape[0], -1, self.num_channels).reshape(-1, 1)
        channel = channels[None, None, :].expand(shifts.shape[0], len(orientations), -1).reshape(-1, 1)
        return torch.cat([torch.full((base.shape[0], 1), level), base, orient, channel], dim=1)

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        if tail_probability <= 0.0:
            raise ValueError("tail_probability must be positive for the infinite Haar prior")
        rows = []
        if self.scaling_mass / self.num_channels >= tail_probability:
            channels = torch.arange(self.num_channels)[:, None]
            rows.append(torch.cat([torch.full((self.num_channels, 1), -1), torch.zeros(self.num_channels, self.domain_dim + 1, dtype=torch.long), channels], dim=1))
        j = 0
        while float(self._wavelet_pmf(torch.tensor(j))) >= tail_probability:
            rows.append(self._level_indices(j))
            j += 1
        return torch.cat(rows, dim=0) if rows else torch.empty((0, self._index_width), dtype=torch.long)

    def sample_indices(self, num_samples: int, with_replacement: bool = True) -> torch.Tensor:
        if not with_replacement:
            raise NotImplementedError("without-replacement sampling is undefined for the infinite Haar prior")
        scaling = torch.rand(num_samples) < self.scaling_mass
        result = torch.zeros(num_samples, self._index_width, dtype=torch.long)
        result[:, 0] = -1
        result[:, -1] = torch.randint(self.num_channels, (num_samples,))
        rows = (~scaling).nonzero(as_tuple=False).flatten()
        if rows.numel():
            level = torch.floor(torch.log1p(-torch.rand(rows.numel())) / math.log(self.level_decay)).long()
            result[rows, 0] = level
            for axis in range(self.domain_dim):
                result[rows, 1 + axis] = (torch.rand(rows.numel()) * (2 ** level)).long()
            result[rows, -2] = torch.randint(1, 2 ** self.domain_dim, (rows.numel(),))
        return result
