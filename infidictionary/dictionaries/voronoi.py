import torch

from .base import InfiDictionary, orthogonal_synthesis


class VoronoiPWC(InfiDictionary):
    """Piecewise-constant dictionary on a regular grid of blocks over ``[0, 1]^d``.

    The domain is split into ``L = resolution**dimension`` equal blocks (the
    Voronoi cells of a regular grid of centers). The *base* atoms are indexed by
    ``(l, c)`` (block ``l``, channel ``c``) and are unit-norm block indicators:

        base_{l,c}(x)[c'] = resolution^{d/2} · 1[x ∈ block l] · 1[c' = c].

    Flattening the multi-index as ``m = l·C + c`` gives ``M = L·C`` orthonormal
    base atoms. An orthogonal synthesis ``W ∈ R^{M × M}`` mixes them:

        atom_a = Σ_m W[a, m] · base_m ,

    learned when ``learn_synthesis=True`` and the identity (the plain block
    basis) otherwise. :meth:`get_atoms` evaluates synthesized atoms with a
    single gather: at a point in block ``l`` only that block's base atoms are
    non-zero, so ``atom_a(x)[c'] = resolution^{d/2} · W[a, l·C + c']``.

    Synthesized atoms are indexed by ``a`` in ``idx[:, 0]`` (``idx[:, 1]`` keeps
    the ``(A, 2)`` layout and is ignored); the prior is harmonic,
    ``p_a ∝ 1 / (a + 1)``.
    """

    def __init__(
        self,
        resolution: int,
        dimension: int,
        num_channels: int = 1,
        learn_synthesis: bool = True,
    ):
        super().__init__()
        if resolution < 1 or dimension < 1 or num_channels < 1:
            raise ValueError("resolution, dimension and num_channels must all be >= 1")
        self.resolution = int(resolution)
        self.domain_dim = int(dimension)
        self.num_channels = int(num_channels)
        self.learn_synthesis = bool(learn_synthesis)
        self._num_cells = self.resolution ** self.domain_dim
        # Each block has volume resolution**(-d), so this makes every indicator unit-norm.
        self._amplitude = self.resolution ** (self.domain_dim / 2)
        self._num_atoms = self._num_cells * self.num_channels

        probs = 1.0 / torch.arange(1, self._num_atoms + 1, dtype=torch.get_default_dtype())
        self.register_buffer("_probs", probs / probs.sum(), persistent=False)
        self._synthesis = orthogonal_synthesis(self._num_atoms, self.learn_synthesis)

    def _assign_cells(self, coords: torch.Tensor) -> torch.Tensor:
        """Block index of each coordinate, ``(N,)``, in row-major (``ij``) order."""
        per_axis = (coords * self.resolution).floor().long().clamp(0, self.resolution - 1)
        cell = torch.zeros(coords.shape[0], dtype=torch.long, device=coords.device)
        for axis in range(self.domain_dim):
            cell = cell * self.resolution + per_axis[:, axis]
        return cell

    def get_atoms(
        self,
        coords: torch.Tensor,  # (N, d)
        idx: torch.Tensor,     # (A, 2)
        synthesis: bool = True,
    ) -> torch.Tensor:         # (A, N, C)
        """Evaluate synthesized atoms, or raw ``(cell, channel)`` block atoms.

        With ``synthesis=True``, ``idx[:, 0]`` selects an output row of the
        synthesis matrix and column 1 is ignored. With ``synthesis=False``,
        the two columns select a raw base atom's cell and output channel.
        """
        device = coords.device
        idx = idx.to(device)
        C = self.num_channels
        A, N = idx.shape[0], coords.shape[0]
        cell = self._assign_cells(coords)  # (N,)

        if not synthesis:
            cell_ids, channels = idx[:, 0].long(), idx[:, 1].long()
            valid = (cell_ids >= 0) & (cell_ids < self._num_cells) & (channels >= 0) & (channels < C)
            active = (cell_ids[:, None] == cell[None, :]) & valid[:, None]
            vals = torch.zeros(A, N, C, dtype=coords.dtype, device=device)
            vals.scatter_(
                2,
                channels.clamp(0, C - 1)[:, None, None].expand(A, N, 1),
                (active.to(coords.dtype) * self._amplitude).unsqueeze(-1),
            )
            return vals

        a = idx[:, 0].long().clamp(0, self._num_atoms - 1)
        W_rows = self._synthesis.weight.to(dtype=coords.dtype, device=device)[a]  # (A, M)
        col = (cell[:, None] * C + torch.arange(C, device=device)[None, :]).reshape(-1)  # (N*C,)
        return W_rows[:, col].reshape(A, N, C) * self._amplitude

    def sample_indices(self, num_samples: int, with_replacement: bool = True) -> torch.Tensor:
        """Sample atom indices ``a`` from the prior; returns ``(num_samples, 2)``."""
        a = torch.multinomial(self._probs, num_samples, replacement=with_replacement)
        return torch.stack([a, torch.zeros_like(a)], dim=-1)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-atom prior ``p_a``; ``0`` for indices outside ``[0, num_atoms)``."""
        a = idx[:, 0]
        valid = (a >= 0) & (a < self._num_atoms)
        probs = self._probs.to(idx.device)
        out = torch.zeros(idx.shape[0], dtype=probs.dtype, device=idx.device)
        out[valid] = probs[a[valid]]
        return out

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Atom indices whose prior is at least ``tail_probability``."""
        atom_ids = torch.arange(self._num_atoms, device=self._probs.device)
        indices = torch.stack([atom_ids, torch.zeros_like(atom_ids)], dim=-1)
        return indices[self._probs >= tail_probability]
