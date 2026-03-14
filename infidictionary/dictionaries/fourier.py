from sklearn.naive_bayes import abstractmethod
import torch
import math
import scipy.stats
import scipy.special
import pytorch_finufft.functional as finufft
from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product

class FourierDictionary(InfiDictionary):
    """Dictionary of real-valued Fourier (trigonometric) atoms on the unit hypercube.

    Each atom is a product of 1-D cosine or sine functions **active in exactly
    one output channel**, forming a complete orthonormal basis of
    ``L^2([0,1]^d, R^C)``.

    Atoms are indexed by ``(k_1, ..., k_d, c)`` where:

    * ``k_i >= 0`` selects the cosine function ``sqrt(2) * cos(2 pi k_i x_i)``
      (constant ``1`` for ``k_i = 0``).
    * ``k_i < 0``  selects the sine   function ``sqrt(2) * sin(2 pi |k_i| x_i)``.
    * ``c in {0, ..., C-1}`` is the output channel index.

    Atom ``(k, c)`` evaluates to ``phi_k(x)`` in channel ``c`` and ``0`` in
    all other channels.  This gives an orthonormal basis:
    ``<phi_{k,c}, phi_{k',c'}> = delta(k,k') delta(c,c')``.

    Index tensors have shape ``(A, domain_dim + 1)`` where the last column is
    the channel index.  Spatial indices are sampled from a Geometric or Zeta
    distribution; the channel index is sampled uniformly over ``{0,...,C-1}``.

    For Geometric: ``P(|k_d|=m) = p*(1-p)^m``.
    For Zeta:      ``P(|k_d|=m) = (m+1)^{-a} / ζ(a)``  (infinite support, a > 1).

    Args:
        domain_dim: Spatial dimension ``d`` of the domain.
        num_channels: Number of output channels ``C``.
        distribution_type: Either ``"geometric"`` or ``"zeta"``.
        distribution_kwargs: Parameters for the chosen distribution.
            For ``"geometric"``: ``{"p": float}`` — success probability.
            For ``"zeta"``: ``{"a": float}`` — power-law exponent (must be > 1).
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        distribution_type: str,
        distribution_kwargs: dict,
    ):
        super().__init__()
        self.domain_dim = domain_dim
        self.num_channels = num_channels
        self.distribution_type = distribution_type
        self.distribution_kwargs = distribution_kwargs
        self.is_orthonormal = True

    # ── Distribution helpers ───────────────────────────────────────────────

    def _sample_1d_magnitudes(self, num_samples: int) -> torch.Tensor:
        """Sample frequency magnitudes (non-negative integers) for one dimension."""
        if self.distribution_type == "geometric":
            p = self.distribution_kwargs["p"]
            return torch.distributions.Geometric(p).sample((num_samples,)).long()
        elif self.distribution_type == "zeta":
            a = self.distribution_kwargs["a"]
            # scipy.stats.zipf(a) is the Zeta distribution; rvs() returns 1,2,3,...
            # subtract 1 to shift support to 0,1,2,...
            samples = scipy.stats.zipf(a).rvs(num_samples) - 1
            return torch.from_numpy(samples).long()
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type!r}")

    def _1d_pmf(self, mag: torch.Tensor) -> torch.Tensor:
        """Return P(|k_d|=m) for each magnitude m (non-negative integer tensor)."""
        if self.distribution_type == "geometric":
            p = self.distribution_kwargs["p"]
            mag_f = mag.float()
            return p * (1.0 - p) ** mag_f
        elif self.distribution_type == "zeta":
            a = self.distribution_kwargs["a"]
            zeta_a = scipy.special.zeta(a, 1)  # Riemann ζ(a)
            mag_f = mag.float()
            return (mag_f + 1.0) ** (-a) / zeta_a
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type!r}")

    def _compute_M_bound(self, tail_probability: float) -> int:
        """Upper bound on spatial frequency magnitude for high-probability indices.

        Returns M such that any index with all |k_d| <= M may have spatial
        probability >= tail_probability; any with some |k_d| > M cannot.
        Pass ``tail_probability * num_channels`` when computing the bound for
        the full joint index (k, c).
        """
        if self.distribution_type == "geometric":
            p = self.distribution_kwargs["p"]
            pd = p ** self.domain_dim
            if pd > 2.0 * tail_probability:
                return int(math.log(pd / (2.0 * tail_probability)) / math.log(1.0 / (1.0 - p)))
            return 0
        elif self.distribution_type == "zeta":
            a = self.distribution_kwargs["a"]
            zeta_a = scipy.special.zeta(a, 1)
            denom = 2.0 * tail_probability * (zeta_a ** self.domain_dim)
            if denom < 1.0:
                return int((1.0 / denom) ** (1.0 / a) - 1)
            return 0
        else:
            raise ValueError(f"Unknown distribution_type: {self.distribution_type!r}")

    # ── Core dictionary methods ────────────────────────────────────────────

    def _get_spatial_atoms(
        self,
        coords: torch.Tensor,       # (N, d)
        spatial_idx: torch.Tensor,  # (A, d)  signed frequency indices
    ) -> torch.Tensor:              # (A, N)  scalar spatial atoms
        """Compute scalar spatial Fourier atoms phi_k(x).

        Returns the product of 1-D trig functions for each atom and coordinate,
        without any channel or normalization factor.
        """
        A, N = spatial_idx.shape[0], coords.shape[0]
        vals = torch.ones((A, N), device=coords.device, dtype=coords.dtype)
        for d in range(self.domain_dim):
            d_idx = spatial_idx[:, d]
            kind = d_idx >= 0  # True → cosine branch, False → sine branch
            freq = torch.abs(d_idx).float()
            kt = 2.0 * math.pi * freq[:, None] * coords[:, d][None, :]  # (A, N)
            cos_component = math.sqrt(2.0) * torch.cos(kt)
            sin_component = math.sqrt(2.0) * torch.sin(kt)
            component = torch.where(
                d_idx[:, None] == 0,
                torch.ones_like(cos_component),
                torch.where(kind[:, None] == 0, cos_component, sin_component),
            )  # (A, N)
            vals *= component
        return vals  # (A, N)

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample Fourier atom indices ``(k_1,...,k_d, c)`` from the joint prior.

        Spatial frequency magnitudes are drawn from the configured distribution;
        the cosine/sine sign is chosen uniformly; the channel ``c`` is drawn
        uniformly from ``{0, ..., C-1}``.

        Args:
            num_samples: Number of indices to sample.

        Returns:
            Integer tensor of shape ``(num_samples, domain_dim + 1)``.
        """
        spatial = torch.zeros((num_samples, self.domain_dim), dtype=torch.long)
        for d in range(self.domain_dim):
            magnitudes = self._sample_1d_magnitudes(num_samples)
            sign = torch.randint(0, 2, (num_samples,))
            spatial[:, d] = torch.where(sign == 0, magnitudes, -magnitudes)
        channels = torch.randint(0, self.num_channels, (num_samples,))
        return torch.cat([spatial, channels.unsqueeze(-1)], dim=-1)  # (num_samples, d+1)

    def get_atoms(
        self,
        coords: torch.Tensor,
        idx: torch.Tensor,  # (A, domain_dim + 1)  last col = channel
    ) -> torch.Tensor:      # (A, N, C)
        """Evaluate Fourier atoms at the given coordinates.

        Atom ``(k, c)`` equals ``phi_k(x)`` in channel ``c`` and ``0`` in all
        other channels, where ``phi_k`` is the product of 1-D trig functions.

        Args:
            coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
            idx: Multi-indices ``(k_1,...,k_d, c)``, shape ``(A, domain_dim+1)``.

        Returns:
            Atom values, shape ``(A, N, C)``.
        """
        spatial_idx = idx[:, :-1]           # (A, d)
        channel_idx = idx[:, -1]            # (A,)
        C = self.num_channels
        A, N = spatial_idx.shape[0], coords.shape[0]

        phi = self._get_spatial_atoms(coords, spatial_idx)  # (A, N)

        vals = torch.zeros((A, N, C), device=coords.device, dtype=coords.dtype)
        # Place phi_k(x) in channel c and leave all other channels at 0.
        c_idx = channel_idx.clamp(0, C - 1)
        vals.scatter_(
            2,
            c_idx[:, None, None].expand(A, N, 1),
            phi.unsqueeze(-1),
        )
        return vals  # (A, N, C)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Prior probability p(k, c) for each multi-index.

        ``p(k, c) = p_spatial(k) / C`` where the spatial probability factors
        across dimensions, and channel is drawn uniformly.

        Signed-index convention: ``P(k_d=0) = P_1d(0)``,
        ``P(k_d=±m) = P_1d(m) / 2`` for m > 0.

        Args:
            idx: Multi-indices ``(k_1,...,k_d, c)``, shape ``(A, domain_dim+1)``.

        Returns:
            Probability tensor of shape ``(A,)``.
        """
        spatial_idx = idx[:, :-1]  # (A, d)
        proba = torch.ones(idx.shape[0], device=idx.device, dtype=torch.float)
        for d in range(self.domain_dim):
            mag = spatial_idx[:, d].abs()
            p_1d = self._1d_pmf(mag)
            p_d = torch.where(mag == 0, p_1d, 0.5 * p_1d)
            proba = proba * p_d
        return proba / self.num_channels  # uniform over channels

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all ``(k, c)`` multi-indices whose prior probability meets the threshold.

        Computes a conservative spatial M-bound (accounting for the 1/C channel
        factor), enumerates the ``[-M, M]^d × {0,...,C-1}`` grid, and filters
        by exact joint probability.

        Args:
            tail_probability: Minimum probability threshold.

        Returns:
            Integer tensor of shape ``(A, domain_dim + 1)``.
        """
        # p(k,c) = p_spatial(k)/C >= tail_probability  ⟺  p_spatial(k) >= C * tail_probability
        M = self._compute_M_bound(tail_probability * self.num_channels)
        vals = torch.arange(-M, M + 1)
        grids = torch.meshgrid(*[vals for _ in range(self.domain_dim)], indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)  # (A_s, d)

        # Cross with all channels
        C = self.num_channels
        A_s = spatial_idx.shape[0]
        channels = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)        # (A_s*C,)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        idx = torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)  # (A_s*C, d+1)

        return idx[self.get_index_pmfs(idx) >= tail_probability]

    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        """Return all ``(k, c)`` indices within a spatial frequency band.

        Constructs the ``(2*num_truncated-1)^d × C`` grid of multi-indices
        ``(k_1,...,k_d, c)`` with ``k_i in {-(num_truncated-1),...,num_truncated-1}``
        and ``c in {0,...,C-1}``.

        Args:
            num_truncated: Half-bandwidth for spatial frequencies.

        Returns:
            Integer tensor of shape ``((2*num_truncated-1)^d * C, domain_dim+1)``.
        """
        freq = torch.arange(-num_truncated + 1, num_truncated)
        grids = torch.meshgrid(*[freq for _ in range(self.domain_dim)], indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)  # (A_s, d)

        C = self.num_channels
        A_s = spatial_idx.shape[0]
        channels = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)        # (A_s*C,)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        return torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)  # (A_s*C, d+1)

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

    def compute_grid_probas(self, nyquist: int) -> torch.Tensor:
        """Compute per-spatial-frequency PMF weights on the ``(2*nyquist+1)^2`` grid.

        Returns the **spatial** probability ``p_spatial(k)`` (without the 1/C
        channel factor).  Used to weight DFT coefficients in
        :meth:`nufft_captured_energy`, which divides by ``num_channels``
        separately.

        Args:
            nyquist: Maximum frequency magnitude.

        Returns:
            Probability tensor of shape ``(2*nyquist+1, 2*nyquist+1)``.
        """
        if self.distribution_type != "geometric":
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        geom_p = self.distribution_kwargs["p"]
        single_tensor = geom_p * (1.0 - geom_p) ** torch.arange(0, nyquist + 1)
        single_tensor = single_tensor / (1 - (1 - geom_p) ** (nyquist + 1))

        weights = torch.ones(*[(nyquist+1) for _ in range(self.domain_dim)], device=single_tensor.device, dtype=single_tensor.dtype)
        for d in range(self.domain_dim):
            shape = [1 for _ in range(self.domain_dim)]
            shape[d] = nyquist + 1
            weights *= single_tensor.view(shape)

        all_probas = torch.zeros((2*nyquist+1, 2*nyquist+1))
        all_probas[nyquist:, nyquist:] += weights
        all_probas[nyquist:, :nyquist+1] += torch.flip(weights, dims=[1])
        all_probas[:nyquist+1, nyquist:] += torch.flip(weights, dims=[0])
        all_probas[:nyquist+1, :nyquist+1] += torch.flip(weights, dims=[0, 1])
        all_probas /= 4

        return all_probas

    def nufft_captured_energy(
        self,
        coords: torch.Tensor,      # (N, d)
        logabsdet: torch.Tensor,   # (N, )
        values: torch.Tensor,      # (B, N, C)
        nyquist: int,
        return_dft: bool = False,
    ) -> torch.Tensor:             # (B, )
        """Estimate captured energy using a Non-Uniform FFT.

        Computes ``E_{k,c}[|<f, phi_{k,c}>|^2]`` efficiently via NUFFT.
        Equivalent to ``(1/C) * sum_k p(k) * sum_c |F_c(k)|^2``, consistent
        with the joint index distribution ``p(k, c) = p_spatial(k) / C``.

        Args:
            coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
            logabsdet: Log absolute value of the measure Jacobian, shape ``(N,)``.
            values: Function values at quadrature points, shape ``(B, N, C)``.
            nyquist: Maximum frequency magnitude.
            return_dft: If ``True``, also return the DFT grid and spatial
                probability tensor.

        Returns:
            Per-function energy estimate of shape ``(B,)``.
        """
        points = (2 * math.pi * coords).transpose(0, 1).contiguous().to(coords.device).to(dtype=torch.float32)
        weights = torch.exp(logabsdet).to(dtype=torch.float32)
        values_in = values.permute(0, 2, 1).contiguous().to(coords.device).to(dtype=torch.complex64)
        values_in = values_in * weights[None, None, :]  # (B, C, N)
        dft = finufft.finufft_type1(
            points=points,
            values=values_in,
            output_shape=(nyquist * 2 + 1, nyquist * 2 + 1),
            modeord=0,
        ).permute(0, 2, 3, 1) / points.shape[1]  # (B, H, W, C)
        all_probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device)  # (H, W)
        # Divide by num_channels: p(k,c) = p_spatial(k) / C
        energy = (dft.abs() ** 2 * all_probas[None, :, :, None]).sum(dim=(1, 2, 3)) / self.num_channels
        return energy if not return_dft else (energy, dft, all_probas)
