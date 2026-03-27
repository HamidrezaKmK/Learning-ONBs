import torch
import math
import pytorch_finufft.functional as finufft
from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product


class FourierDictionary(InfiDictionary):
    """Dictionary of real-valued Fourier (trigonometric) atoms on the unit hypercube.

    Each atom is a product of 1-D cosine or sine functions active in exactly
    one output channel, forming a complete orthonormal basis of L²([0,1]^d, R^C).

    Atoms are indexed by ``(k_1, ..., k_d, c)`` where:

    * ``k_i >= 0`` selects ``sqrt(2) * cos(2π k_i x_i)`` (constant 1 for ``k_i=0``).
    * ``k_i < 0``  selects ``sqrt(2) * sin(2π |k_i| x_i)``.
    * ``c in {0,...,C-1}`` is the output channel index.

    **PMF — L∞ shell distribution.**
    Rather than independent per-dimension geometric draws, the prior is defined
    by the L∞ shell radius ``m = max(|k_1|,...,|k_d|)``:

        P(k, c) = p·(1-p)^m / shell_size(m) / C

    where ``shell_size(m)`` is the number of integer vectors in Z^d with
    L∞ norm exactly ``m``.  All indices in the same L∞ shell are therefore
    **equally probable**, e.g. for d=2 the indices (3,0), (3,1), (3,3),
    (0,-3), (-2,3), … all share the same PMF value.

    Sampling draws ``m ~ Geometric(p)``, then picks a uniform element from
    shell ``m`` via rejection sampling on the [-m,m]^d box.

    Args:
        domain_dim:   Spatial dimension ``d``.
        num_channels: Number of output channels ``C``.
        p:            Success probability for the geometric shell distribution.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        p: float = 0.3,
    ):
        super().__init__()
        self.domain_dim   = domain_dim
        self.num_channels = num_channels
        self.p            = p
        self.is_orthonormal = True

    # ── Shell helpers ──────────────────────────────────────────────────────────

    def _shell_size(self, m: torch.Tensor) -> torch.Tensor:
        """Number of integer vectors in Z^d with L∞ norm exactly m.

        shell_size(0) = 1
        shell_size(m) = (2m+1)^d − (2m−1)^d   for m ≥ 1
        """
        d = self.domain_dim
        return torch.where(
            m == 0,
            torch.ones_like(m, dtype=torch.float),
            ((2 * m + 1).pow(d) - (2 * m - 1).pow(d)).float(),
        )

    def _sample_from_shells(self, shells: torch.Tensor) -> torch.Tensor:
        """Sample one spatial index uniformly from L∞ shell(m) for each m.

        Uses rejection sampling on the [-m,m]^d box; for d=2 the expected
        number of iterations is ≤ 2 for all practically occurring shell radii.

        Args:
            shells: (S,) non-negative integer shell indices.
        Returns:
            (S, d) integer tensor of spatial indices.
        """
        S = shells.shape[0]
        d = self.domain_dim
        result = torch.zeros(S, d, dtype=torch.long)
        done   = torch.zeros(S, dtype=torch.bool)

        while not done.all():
            m = shells   # (S,)
            # Sample ki uniformly from {-m, ..., m} for each dimension
            cands = torch.zeros(S, d, dtype=torch.long)
            for dim in range(d):
                r = torch.rand(S)
                cands[:, dim] = (r * (2 * m.float() + 1)).long() - m
            # Accept samples that land exactly on the shell (max |ki| == m)
            on_shell = cands.abs().amax(dim=-1) == m
            accept   = on_shell & ~done
            result[accept] = cands[accept]
            done = done | accept

        return result

    # ── Core dictionary methods ────────────────────────────────────────────────

    def _get_spatial_atoms(
        self,
        coords: torch.Tensor,       # (N, d)
        spatial_idx: torch.Tensor,  # (A, d) signed frequency indices
    ) -> torch.Tensor:              # (A, N)
        """Compute scalar Fourier atoms phi_k(x)."""
        A, N = spatial_idx.shape[0], coords.shape[0]
        vals = torch.ones((A, N), device=coords.device, dtype=coords.dtype)
        for d in range(self.domain_dim):
            d_idx = spatial_idx[:, d]
            freq  = torch.abs(d_idx).float()
            kt    = 2.0 * math.pi * freq[:, None] * coords[:, d][None, :]  # (A, N)
            cos_c = math.sqrt(2.0) * torch.cos(kt)
            sin_c = math.sqrt(2.0) * torch.sin(kt)
            component = torch.where(
                d_idx[:, None] == 0,
                torch.ones_like(cos_c),
                torch.where(d_idx[:, None] > 0, cos_c, sin_c),
            )
            vals *= component
        return vals  # (A, N)

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample atom indices ``(k_1,...,k_d, c)`` from the L∞ shell prior.

        1. Draw shell radius ``m ~ Geometric(p)``.
        2. Sample spatial index uniformly from shell ``m``.
        3. Draw channel ``c ~ Uniform{0,...,C-1}``.

        Args:
            num_samples: Number of indices to draw.
        Returns:
            Integer tensor of shape ``(num_samples, domain_dim + 1)``.
        """
        shells   = torch.distributions.Geometric(self.p).sample((num_samples,)).long()
        spatial  = self._sample_from_shells(shells)           # (S, d)
        channels = torch.randint(0, self.num_channels, (num_samples,))
        return torch.cat([spatial, channels.unsqueeze(-1)], dim=-1)

    def get_atoms(
        self,
        coords: torch.Tensor,  # (N, d)
        idx: torch.Tensor,     # (A, d+1)  last col = channel
    ) -> torch.Tensor:         # (A, N, C)
        """Evaluate Fourier atoms at the given coordinates.

        Atom ``(k, c)`` equals ``phi_k(x)`` in channel ``c`` and 0 elsewhere.
        """
        spatial_idx = idx[:, :-1]          # (A, d)
        channel_idx = idx[:, -1]           # (A,)
        C = self.num_channels
        A, N = spatial_idx.shape[0], coords.shape[0]

        phi  = self._get_spatial_atoms(coords, spatial_idx)  # (A, N)
        vals = torch.zeros((A, N, C), device=coords.device, dtype=coords.dtype)
        c_idx = channel_idx.clamp(0, C - 1)
        vals.scatter_(2, c_idx[:, None, None].expand(A, N, 1), phi.unsqueeze(-1))
        return vals  # (A, N, C)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Prior probability for each multi-index under the L∞ shell distribution.

        P(k, c) = p·(1-p)^m / shell_size(m) / C,   m = max_i |k_i|

        Args:
            idx: Multi-indices ``(k_1,...,k_d, c)``, shape ``(A, d+1)``.
        Returns:
            Probability tensor of shape ``(A,)``.
        """
        spatial_idx = idx[:, :-1]                            # (A, d)
        m        = spatial_idx.abs().amax(dim=-1)            # (A,)
        p_shell  = self.p * (1.0 - self.p) ** m.float()     # P(shell = m)
        shell_sz = self._shell_size(m)                       # (A,)
        return p_shell / shell_sz / self.num_channels

    def _compute_M_bound(self, tail_probability: float) -> int:
        """Largest shell radius m whose indices can have PMF >= tail_probability / C.

        Since P(k,c) ≤ p·(1-p)^m / C, once p·(1-p)^m < tail_probability the
        whole shell is below threshold.

        Args:
            tail_probability: Adjusted threshold (already multiplied by C by caller).
        Returns:
            Non-negative integer M.
        """
        p = self.p
        if tail_probability >= p:
            return 0
        return int(math.log(tail_probability / p) / math.log(1.0 - p))

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all ``(k, c)`` multi-indices with prior PMF >= tail_probability.

        Args:
            tail_probability: Minimum probability threshold.
        Returns:
            Integer tensor of shape ``(A, domain_dim + 1)``.
        """
        M = self._compute_M_bound(tail_probability * self.num_channels)
        vals = torch.arange(-M, M + 1)
        grids = torch.meshgrid(*[vals] * self.domain_dim, indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)  # (A_s, d)

        C   = self.num_channels
        A_s = spatial_idx.shape[0]
        channels    = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        idx = torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)  # (A_s*C, d+1)

        return idx[self.get_index_pmfs(idx) >= tail_probability]

    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        """Return all ``(k, c)`` indices within the L∞ ball of radius num_truncated-1.

        Args:
            num_truncated: Half-bandwidth; frequencies in {-(num_truncated-1),...,num_truncated-1}.
        Returns:
            Integer tensor of shape ``((2*num_truncated-1)^d * C, domain_dim+1)``.
        """
        freq = torch.arange(-num_truncated + 1, num_truncated)
        grids = torch.meshgrid(*[freq] * self.domain_dim, indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)

        C   = self.num_channels
        A_s = spatial_idx.shape[0]
        channels    = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        return torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)

    # Low-variance but truncated alternative (nyquist=4, no tail MC):
    # def monte_carlo_captured_energy(self, coords, logabsdet, values, num_tail_samples, tail_probability=1e-4):
    #     idx = self.get_truncated_indices(5).to(coords.device)  # (A*C, d+1)
    #     atoms = self.get_atoms(coords, idx)  # (A*C, N, C)
    #     coefficients = pairwise_inner_product(values, atoms, logabsdet)  # (B, A*C)
    #     nyquist = int(idx[:, :-1].abs().max().item())  # spatial nyquist = 4
    #     probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device)  # (2*nyquist+1)^2
    #     # Map each (k1, k2, c) to its spatial probability / C
    #     spatial_probas = probas[idx[:, 0] + nyquist, idx[:, 1] + nyquist] / self.num_channels
    #     return (coefficients ** 2 * spatial_probas[None, :]).sum(dim=-1)  # (B,)

    # def compute_grid_probas(self, nyquist: int) -> torch.Tensor:
    #     """Compute per-spatial-frequency PMF weights on the ``(2*nyquist+1)^2`` grid.

    #     Returns the **spatial** probability ``p_spatial(k)`` (without the 1/C
    #     channel factor).  Used to weight DFT coefficients in
    #     :meth:`nufft_captured_energy`, which divides by ``num_channels``
    #     separately.

    #     Args:
    #         nyquist: Maximum frequency magnitude.

    #     Returns:
    #         Probability tensor of shape ``(2*nyquist+1, 2*nyquist+1)``.
    #     """
    #     if self.distribution_type != "geometric":
    #         raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
    #     geom_p = self.distribution_kwargs["p"]
    #     single_tensor = geom_p * (1.0 - geom_p) ** torch.arange(0, nyquist + 1)
    #     single_tensor = single_tensor / (1 - (1 - geom_p) ** (nyquist + 1))

    #     weights = torch.ones(*[(nyquist+1) for _ in range(self.domain_dim)], device=single_tensor.device, dtype=single_tensor.dtype)
    #     for d in range(self.domain_dim):
    #         shape = [1 for _ in range(self.domain_dim)]
    #         shape[d] = nyquist + 1
    #         weights *= single_tensor.view(shape)

    #     all_probas = torch.zeros((2*nyquist+1, 2*nyquist+1))
    #     all_probas[nyquist:, nyquist:] += weights
    #     all_probas[nyquist:, :nyquist+1] += torch.flip(weights, dims=[1])
    #     all_probas[:nyquist+1, nyquist:] += torch.flip(weights, dims=[0])
    #     all_probas[:nyquist+1, :nyquist+1] += torch.flip(weights, dims=[0, 1])
    #     all_probas /= 4

    #     return all_probas

    # def nufft_captured_energy(
    #     self,
    #     coords: torch.Tensor,      # (N, d)
    #     logabsdet: torch.Tensor,   # (N, )
    #     values: torch.Tensor,      # (B, N, C)
    #     nyquist: int,
    #     return_dft: bool = False,
    # ) -> torch.Tensor:             # (B, )
    #     """Estimate captured energy using a Non-Uniform FFT.

    #     Computes ``E_{k,c}[|<f, phi_{k,c}>|^2]`` efficiently via NUFFT.
    #     Equivalent to ``(1/C) * sum_k p(k) * sum_c |F_c(k)|^2``, consistent
    #     with the joint index distribution ``p(k, c) = p_spatial(k) / C``.

    #     Args:
    #         coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
    #         logabsdet: Log absolute value of the measure Jacobian, shape ``(N,)``.
    #         values: Function values at quadrature points, shape ``(B, N, C)``.
    #         nyquist: Maximum frequency magnitude.
    #         return_dft: If ``True``, also return the DFT grid and spatial
    #             probability tensor.

    #     Returns:
    #         Per-function energy estimate of shape ``(B,)``.
    #     """
    #     points = (2 * math.pi * coords).transpose(0, 1).contiguous().to(coords.device).to(dtype=torch.float32)
    #     weights = torch.exp(logabsdet).to(dtype=torch.float32)
    #     values_in = values.permute(0, 2, 1).contiguous().to(coords.device).to(dtype=torch.complex64)
    #     values_in = values_in * weights[None, None, :]  # (B, C, N)
    #     dft = finufft.finufft_type1(
    #         points=points,
    #         values=values_in,
    #         output_shape=(nyquist * 2 + 1, nyquist * 2 + 1),
    #         modeord=0,
    #     ).permute(0, 2, 3, 1) / points.shape[1]  # (B, H, W, C)
    #     all_probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device)  # (H, W)
    #     # Divide by num_channels: p(k,c) = p_spatial(k) / C
    #     energy = (dft.abs() ** 2 * all_probas[None, :, :, None]).sum(dim=(1, 2, 3)) / self.num_channels
    #     return energy if not return_dft else (energy, dft, all_probas)
