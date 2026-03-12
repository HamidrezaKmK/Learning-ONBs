from sklearn.naive_bayes import abstractmethod
import torch
import math
import pytorch_finufft.functional as finufft
from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product

class FourierDictionary(InfiDictionary):
    """Dictionary of real-valued Fourier (trigonometric) atoms on the unit hypercube.

    Each atom is a product of 1-D cosine or sine functions.  Atoms are indexed
    by a ``d``-dimensional integer vector ``k = (k_1, ..., k_d)``:

    * ``k_i >= 0`` selects the cosine atom ``sqrt(2) * cos(2 pi k_i x_i)``
      along dimension ``i`` (with the constant function ``1`` for ``k_i = 0``).
    * ``k_i < 0``  selects the sine   atom ``sqrt(2) * sin(2 pi |k_i| x_i)``.

    The resulting atoms form an orthonormal basis of ``L^2([0,1]^d, R^C)``
    (with channel-wise normalization by ``1/sqrt(C)``).

    Atom indices are sampled from a geometric distribution over frequencies,
    controlled by ``index_geom_p``, with the cosine/sine sign chosen uniformly
    at random.

    Args:
        domain_dim: Spatial dimension ``d`` of the domain.
        num_channels: Number of output channels ``C``.
        index_geom_p: Success probability of the Geometric distribution used
            to sample frequency magnitudes.  Larger values concentrate mass on
            lower frequencies.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        index_geom_p: float,
    ):
        super().__init__()
        self.domain_dim = domain_dim
        self.index_geom_p = index_geom_p
        self.num_channels = num_channels
        self.is_orthonormal = True

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample Fourier atom indices from the geometric prior.

        For each dimension, the frequency magnitude is drawn from a Geometric
        distribution with parameter ``index_geom_p``, and the sign (cosine vs.
        sine) is chosen uniformly: positive index → cosine, negative → sine.

        Args:
            num_samples: Number of indices to sample.

        Returns:
            Integer tensor of shape ``(num_samples, domain_dim)``.
        """
        idx = torch.zeros((num_samples, self.domain_dim), dtype=torch.long)
        for d in range(self.domain_dim):
            # sample num_samples from a geometric distribution with p = self.index_geom_p
            geom_samples = torch.distributions.Geometric(self.index_geom_p).sample((num_samples,))
            # randomly assign half of the samples to be sine and half to be cosine
            kind_samples = torch.randint(0, 2, (num_samples,))
            idx[:, d] = torch.where(
                kind_samples == 0,
                geom_samples,
                - geom_samples,
            )
        return idx

    def get_atoms(
        self,
        coords: torch.Tensor,
        idx: torch.Tensor, # (A, domain_dim)
    ):
        """Evaluate Fourier atoms at the given coordinates.

        Each atom is the tensor product of 1-D basis functions across dimensions:
        ``phi_k(x) = (1/sqrt(C)) * prod_i f_{k_i}(x_i)``, where
        ``f_0(t) = 1``, ``f_{k>0}(t) = sqrt(2) cos(2 pi k t)``,
        ``f_{k<0}(t) = sqrt(2) sin(2 pi |k| t)``.

        Args:
            coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
            idx: Signed frequency multi-indices, shape ``(A, domain_dim)``.

        Returns:
            Atom values, shape ``(A, N, C)``.
        """
        vals = torch.ones((idx.shape[0], coords.shape[0], self.num_channels), device=coords.device, dtype=coords.dtype)
        for d in range(self.domain_dim):
            d_idx = idx[:, d]
            kind = d_idx >= 0 # 1 for cosine, 0 for sine
            freq = torch.abs(d_idx).float() # (A, )
            kt = 2.0 * math.pi * freq[:, None] * coords[:, d][None, :]   # (N_idx, N_coords)
            cos_component = math.sqrt(2.0) * torch.cos(kt)
            sin_component = math.sqrt(2.0) * torch.sin(kt)
            component = torch.where(
                d_idx[:, None] == 0,
                torch.ones_like(cos_component),
                torch.where(kind[:, None] == 0, cos_component, sin_component)
            ) # (A, N_coords)
            vals *= component[:, :, None] # (A, N_coords, 1) broadcast to (A, N_coords, C)
        return vals / math.sqrt(self.num_channels) # shape (A, N_coords)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Prior probability p(k) for each signed frequency multi-index.

        For each dimension ``d``:
            ``P(k_d = 0)   = p``
            ``P(k_d = ±m)  = p * (1-p)^m / 2``  for ``m > 0``,
        where ``p = index_geom_p``.  The joint probability is the product
        across dimensions.

        Args:
            idx: Signed frequency multi-indices, shape ``(A, domain_dim)``.

        Returns:
            Probability tensor of shape ``(A,)``.
        """
        p = self.index_geom_p
        proba = torch.ones(idx.shape[0], device=idx.device, dtype=torch.float)
        for d in range(self.domain_dim):
            k_d = idx[:, d].float()
            mag = k_d.abs()
            p_d = torch.where(
                mag == 0,
                torch.full_like(mag, p),
                0.5 * p * (1.0 - p) ** mag,
            )
            proba = proba * p_d
        return proba

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all multi-indices whose prior probability meets the threshold.

        Derives a tight bound ``M`` on the per-dimension frequency magnitude
        that any high-probability index can have, enumerates the
        ``[-M, M]^d`` grid, and filters by exact probability.

        The bound follows from requiring
        ``p^d * (1-p)^M / 2 >= tail_probability``, giving
        ``M = floor( log(p^d / (2 * tail_probability)) / log(1/(1-p)) )``.

        Args:
            tail_probability: Minimum probability threshold.

        Returns:
            Integer tensor of shape ``(A, domain_dim)``.
        """
        p = self.index_geom_p
        pd = p ** self.domain_dim
        if pd > 2.0 * tail_probability:
            M = int(math.log(pd / (2.0 * tail_probability)) / math.log(1.0 / (1.0 - p)))
        else:
            M = 0

        vals = torch.arange(-M, M + 1)
        grids = torch.meshgrid(*[vals for _ in range(self.domain_dim)], indexing='ij')
        idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)
        return idx[self.get_index_pmfs(idx) >= tail_probability]

    # Low-variance but truncated alternative (nyquist=4, no tail MC):
    # def monte_carlo_captured_energy(self, coords, logabsdet, values, num_tail_samples, tail_probability=1e-4):
    #     idx = self.get_truncated_indices(5).to(coords.device)  # (A, 2), unique by construction
    #     atoms = self.get_atoms(coords, idx)  # (A, N, C)
    #     coefficients = pairwise_inner_product(values, atoms, logabsdet)  # (B, A)
    #     nyquist = int(idx.abs().max().item())  # = 4
    #     probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device)  # (2*nyquist+1, 2*nyquist+1)
    #     atom_probas = probas[idx[:, 0] + nyquist, idx[:, 1] + nyquist]  # (A,)
    #     return (coefficients ** 2 * atom_probas[None, :]).sum(dim=-1)  # (B,)

    def compute_grid_probas(self, nyquist: int) -> torch.Tensor:
        """Compute per-frequency PMF weights on the ``(2*nyquist+1)^2`` DFT grid.

        Returns a 2-D probability tensor whose entry at ``(k_1, k_2)`` gives the
        probability ``p(k)`` under the (truncated) geometric prior, accounting
        for the fact that cosine and sine share the same frequency magnitude.
        Used to weight DFT coefficients in :meth:`nufft_captured_energy`.

        The single-dimensional probabilities are truncated geometric PMFs,
        normalized to sum to 1 over ``{0, ..., nyquist}``.  The joint
        probability is their tensor product, and each signed-frequency cell
        receives an equal share of the joint mass.

        Args:
            nyquist: Maximum frequency magnitude; the grid spans frequencies
                ``-nyquist, ..., 0, ..., nyquist`` in each dimension.

        Returns:
            Probability tensor of shape ``(2*nyquist+1, 2*nyquist+1)``.
        """
        single_tensor = self.index_geom_p * (1.0 - self.index_geom_p) ** torch.arange(0, nyquist + 1)
        single_tensor = single_tensor / (1 - (1 - self.index_geom_p) ** (nyquist + 1))

        # now do a tensor product to get d_dimensional weights
        weights = torch.ones(*[(nyquist+1) for _ in range(self.domain_dim)], device=single_tensor.device, dtype=single_tensor.dtype) # (nyquist, nyquist, ...)
        for d in range(self.domain_dim):
            shape = [1 for _ in range(self.domain_dim)]
            shape[d] = nyquist + 1
            weights *= single_tensor.view(shape) # (1, 1, ..., nyquist+1, ..., 1) where the nyquist+1 is in the d-th dimension

        all_probas = torch.zeros((2*nyquist+1, 2*nyquist+1))
        all_probas[nyquist:, nyquist:] += weights
        all_probas[nyquist:, :nyquist+1] += torch.flip(weights, dims=[1])
        all_probas[:nyquist+1, nyquist:] += torch.flip(weights, dims=[0])
        all_probas[:nyquist+1, :nyquist+1] += torch.flip(weights, dims=[0, 1])
        all_probas /= 4

        return all_probas

    def nufft_captured_energy(
        self,
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        nyquist: int,
        return_dft: bool = False,
    ) -> torch.Tensor: # (B, )
        """Estimate captured energy using a Non-Uniform FFT.

        Computes all Fourier coefficients up to the Nyquist frequency in one
        efficient NUFFT pass, then weights their squared magnitudes by the
        per-frequency PMF to obtain
        ``E_k[|<f, phi_k>|^2]``.

        Args:
            coords: Quadrature points in ``[0, 1)^d``, shape ``(N, d)``.
            logabsdet: Log absolute value of the measure Jacobian, shape ``(N,)``.
            values: Function values at quadrature points, shape ``(B, N, C)``.
            nyquist: Maximum frequency magnitude; determines the DFT grid size
                ``(2*nyquist+1)^d``.
            return_dft: If ``True``, also return the DFT grid and probability
                tensor (useful for inspection / debugging).

        Returns:
            Per-function energy estimate of shape ``(B,)``, or a tuple
            ``(energy, dft, all_probas)`` when ``return_dft=True``.
        """
        # (d, N), scaled to [-π, π] as finufft expects
        points = (2 * math.pi * coords).transpose(0, 1).contiguous().to(coords.device).to(dtype=torch.float32)
        weights = torch.exp(logabsdet).to(dtype=torch.float32)  # (N,)
        values = values.permute(0, 2, 1).contiguous().to(coords.device).to(dtype=torch.complex64)
        values = values * weights[None, None, :]  # (B, C, N) — apply measure weights
        dft = finufft.finufft_type1(
            points=points,  # (d, N) as expected by finufft
            values=values,  # (B, C, N)
            output_shape=(nyquist * 2 + 1, nyquist * 2 + 1),
            modeord=0,
        ).permute(0, 2, 3, 1) / points.shape[1] # (B, H, W, C)
        all_probas = self.compute_grid_probas(nyquist=nyquist).to(coords.device) # (2 * nyquist + 1, 2 * nyquist + 1)
        energy = (dft.abs() ** 2 * all_probas[None, :, :, None]).sum(dim=(1, 2, 3)) # (B, )
        # energy_normalized = energy / nyquist
        return energy if not return_dft else (energy, dft, all_probas)

    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        """Return all Fourier multi-indices within a frequency band.

        Constructs the full ``(2*num_truncated - 1)^d`` grid of signed integer
        indices in ``{-num_truncated+1, ..., num_truncated-1}^d``.

        Args:
            num_truncated: Half-bandwidth; the grid spans frequencies
                ``-(num_truncated-1)`` to ``num_truncated-1`` in each dimension.

        Returns:
            Integer tensor of shape ``((2*num_truncated-1)^d, domain_dim)``.
        """
        idx = torch.arange(-num_truncated + 1, num_truncated)
        grid = torch.stack(
            torch.meshgrid(*[idx for _ in range(self.domain_dim)], indexing='ij'),
            dim=-1,
        ).view(-1, self.domain_dim)
        return grid
