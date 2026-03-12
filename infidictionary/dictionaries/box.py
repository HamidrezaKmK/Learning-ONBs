
import torch
import math

from .base import InfiDictionary

class BoxDictionary(InfiDictionary):
    """Dictionary of axis-aligned box (indicator) functions on the unit hypercube.

    Each atom is a product of 1-D box functions, one per spatial dimension.
    The domain ``[0, 1)^d`` is partitioned into ``resolution^d`` equal cells;
    each multi-index ``(k_1, ..., k_d)`` with ``k_i in {0, ..., resolution-1}``
    identifies the cell ``[k_i/resolution, (k_i+1)/resolution)`` along dimension
    ``i``.  Atoms are normalized to form an orthonormal basis with respect to
    the uniform Lebesgue measure.

    # TODO: switch this with Haar wavelet basis

    Args:
        domain_dim: Spatial dimension ``d`` of the domain.
        num_channels: Number of output channels ``C``.
        resolution: Number of cells per dimension; determines the finest
            representable scale.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        resolution: int,
    ):
        super().__init__()
        self.domain_dim = domain_dim
        self.num_channels = num_channels
        self.resolution = resolution
        self.is_orthonormal = True

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample atom indices uniformly over the ``resolution^d`` grid cells.

        Each coordinate of the multi-index is drawn independently and uniformly
        from ``{0, ..., resolution - 1}``.

        Args:
            num_samples: Number of indices to sample.

        Returns:
            Integer tensor of shape ``(num_samples, domain_dim)``.
        """
        idx = torch.zeros((num_samples, self.domain_dim), dtype=torch.long)
        for d in range(self.domain_dim):
            idx[:, d] = torch.randint(0, self.resolution, (num_samples,))
        return idx

    def get_atoms(self, coords: torch.Tensor, idx: torch.Tensor):
        """Evaluate box atoms at the given coordinates.

        Atom ``k = (k_1, ..., k_d)`` equals
        ``resolution^(d/2) / sqrt(C)`` inside the cell
        ``prod_i [k_i/resolution, (k_i+1)/resolution)`` and zero elsewhere.
        The normalization factor ensures unit L2 norm under the uniform measure.

        Args:
            coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
            idx: Cell multi-indices, shape ``(A, domain_dim)``.

        Returns:
            Atom values, shape ``(A, N, C)``.
        """
        vals = torch.ones(
            (idx.shape[0], coords.shape[0], self.num_channels),
            device=coords.device, dtype=coords.dtype,
        )
        for d in range(self.domain_dim):
            d_idx = idx[:, d] # (A, )
            # create a box function centered at d_idx with width 1
            box_component = torch.where(
                (coords[:, d] * self.resolution >= d_idx[:, None]) & (coords[:, d] * self.resolution < d_idx[:, None] + 1),
                torch.ones_like(coords[:, d]),
                torch.zeros_like(coords[:, d]),
            ) # (A, N_coords)
            vals *= box_component[:, :, None] # (A, N_coords, 1) broadcast to (A, N_coords, C)
        return vals * math.sqrt(self.resolution ** self.domain_dim) / math.sqrt(self.num_channels) # shape (A, N_coords)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the uniform prior probability for each index.

        All ``resolution^d`` atoms are equally likely, so every index receives
        probability ``1 / resolution^d``.

        Args:
            idx: Cell multi-indices, shape ``(A, domain_dim)``.

        Returns:
            Probability tensor of shape ``(A,)``, all equal to
            ``1 / resolution^domain_dim``.
        """
        uniform_pmf = 1.0 / (self.resolution ** self.domain_dim)
        return torch.full((idx.shape[0],), uniform_pmf, dtype=torch.float)

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all indices if the uniform PMF meets the threshold, else none.

        Since all atoms are equally likely with probability ``1 / resolution^d``,
        either all indices exceed the threshold or none do.

        Args:
            tail_probability: Minimum probability threshold.

        Returns:
            All ``resolution^d`` indices if ``1/resolution^d >= tail_probability``,
            otherwise an empty tensor of shape ``(0, domain_dim)``.
        """
        if 1.0 / (self.resolution ** self.domain_dim) >= tail_probability:
            return self.get_truncated_indices(self.resolution)
        return torch.zeros((0, self.domain_dim), dtype=torch.long)

    def get_truncated_indices(self, num_truncated: int):
        """Return the first ``num_truncated^d`` box indices on a regular grid.

        Indices are formed by taking the Cartesian product
        ``{0, ..., num_truncated-1}^d``, clamped so that ``num_truncated``
        does not exceed ``resolution``.

        Args:
            num_truncated: Number of cells per dimension to include.

        Returns:
            Integer tensor of shape ``(num_truncated^d, domain_dim)``.
        """
        num_truncated = min(num_truncated, self.resolution)
        grids = [torch.arange(num_truncated) for _ in range(self.domain_dim)]
        return torch.cartesian_prod(*grids)
