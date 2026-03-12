from abc import ABC, abstractmethod
from typing import Literal
import math

import torch

from infidictionary.utils import pairwise_inner_product

class InfiDictionary(ABC):
    """Abstract base class for infinite-dimensional dictionaries.

    A dictionary is a (possibly infinite) collection of atoms — scalar- or
    vector-valued functions defined on a continuous domain.  Subclasses
    implement specific families of atoms (e.g. Fourier, box/indicator) and
    provide the core operations needed for sparse approximation and energy
    estimation in function space.

    Atoms are evaluated at a finite set of *coordinates* (quadrature points)
    and are identified by integer *indices* whose meaning is dictionary-specific.

    Shape conventions used throughout:
        B  — batch size (number of functions)
        N  — number of quadrature / sample points
        d  — spatial dimension of the domain
        C  — number of channels (output dimension of each function)
        A  — number of atoms
    """

    @abstractmethod
    def get_atoms(
        self,
        coords: torch.Tensor, # (N, d)
        idx: torch.Tensor, # (A, ...)
    ) -> torch.Tensor: # (A, N, C)
        """Evaluate dictionary atoms at the given coordinates.

        Args:
            coords: Quadrature / sample points, shape ``(N, d)``.
            idx: Integer indices selecting which atoms to evaluate,
                shape ``(A, ...)``.  The inner dimensions are
                dictionary-specific (e.g. ``(A, d)`` for multi-index atoms).

        Returns:
            Atom values at each coordinate, shape ``(A, N, C)``.
        """
        raise NotImplementedError("Subclasses must implement this method if needed")

    @abstractmethod
    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Draw random atom indices according to the dictionary's sampling distribution.

        Args:
            num_samples: Number of atom indices to sample.

        Returns:
            Sampled indices, shape ``(num_samples, ...)``.
        """
        raise NotImplementedError("Subclasses must implement this method if needed")

    @abstractmethod
    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the prior probability p(k) for each multi-index.

        Args:
            idx: Integer multi-indices, shape ``(A, ...)``.

        Returns:
            Probability tensor of shape ``(A,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all indices whose prior probability is at least ``tail_probability``.

        Implementations should enumerate the (finite) set of indices ``k``
        satisfying ``p(k) >= tail_probability`` and return them as a tensor.

        Args:
            tail_probability: Minimum probability threshold.

        Returns:
            Integer tensor of shape ``(A, ...)``.
        """
        raise NotImplementedError

    def monte_carlo_captured_energy(
        self,
        coords: torch.Tensor,       # (N, d)
        logabsdet: torch.Tensor,    # (N,)
        values: torch.Tensor,       # (B, N, C)
        num_tail_samples: int,
        tail_probability: float = 1e-4,
    ) -> torch.Tensor:              # (B,)
        """Estimate captured energy with a stratified low-variance estimator.

        Splits ``E_k[|<f, phi_k>|^2]`` into two strata:

        * **Exact stratum** – all indices with ``p(k) >= tail_probability``,
          computed analytically (zero variance).
        * **MC tail stratum** – remaining indices, estimated by drawing
          ``num_tail_samples`` from the full prior and discarding those already
          covered by the exact stratum.

        Both strata are unbiased; together they recover the full expectation.
        Requires subclasses to implement :meth:`get_index_pmfs`,
        :meth:`get_high_probability_indices`, :meth:`get_atoms`, and
        :meth:`sample_indices`.

        Args:
            coords: Quadrature points, shape ``(N, d)``.
            logabsdet: Log-absolute-value of the measure Jacobian, shape ``(N,)``.
            values: Function values at quadrature points, shape ``(B, N, C)``.
            num_tail_samples: Number of index draws for the MC tail estimate.
            tail_probability: Probability threshold separating the two strata
                (default ``1e-4``).

        Returns:
            Per-function energy estimate, shape ``(B,)``.
        """
        # ── exact high-probability stratum ────────────────────────────────────
        idx_exact = self.get_high_probability_indices(tail_probability).to(coords.device)
        atoms_exact = self.get_atoms(coords, idx_exact)                          # (A_exact, N, C)
        coeffs_exact = pairwise_inner_product(values, atoms_exact, logabsdet)    # (B, A_exact)
        probas_exact = self.get_index_pmfs(idx_exact)                            # (A_exact,)
        energy_exact = (coeffs_exact ** 2 * probas_exact[None, :]).sum(dim=-1)  # (B,)

        # ── MC tail stratum ───────────────────────────────────────────────────
        idx_all = self.sample_indices(num_tail_samples).to(coords.device)        # (S, d)
        in_exact = (idx_all[:, None, :] == idx_exact[None, :, :]).all(dim=-1).any(dim=-1)
        idx_tail = idx_all[~in_exact]                                            # (S_tail, d)

        if idx_tail.shape[0] == 0:
            return energy_exact

        idx_tail_u, idx_tail_c = torch.unique(idx_tail, return_counts=True, dim=0)
        atoms_tail = self.get_atoms(coords, idx_tail_u)                          # (A_tail, N, C)
        coeffs_tail = pairwise_inner_product(values, atoms_tail, logabsdet)      # (B, A_tail)
        energy_tail = (coeffs_tail ** 2 * idx_tail_c[None, :]).sum(dim=-1) / num_tail_samples

        return energy_exact + energy_tail

    @abstractmethod
    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        """Return a deterministic set of indices for a truncated dictionary.

        Used to build a finite sub-dictionary for reconstruction or analysis.

        Args:
            num_truncated: Controls the size of the truncated set
                (interpretation is dictionary-specific, e.g. number of
                frequency shells, number of resolution levels).

        Returns:
            Indices tensor, shape ``(A, ...)``.
        """
        raise NotImplementedError("Subclasses must implement this method if needed")

    def get_reconstructions(
        self,
        coords: torch.Tensor, # (N, d)
        functions: torch.Tensor, # (B, N, C)
        truncation_factor: int,
    ):
        """
        Compute the reconstruction of each of the functions according to the
        indices in the truncation_factor.
        Returns:
            reconstructions: list of torch.Tensor, each of shape (B, N, C)
        """
        device = coords.device
        atom_indices = self.get_truncated_indices(truncation_factor).to(device)
        dictionary_values = self.get_atoms(
            coords,
            atom_indices,
        )  # shape (A, N, C)
        c = pairwise_inner_product(
            functions,
            dictionary_values,
        ) # shape (B, A)
        recon = c @ dictionary_values.view(dictionary_values.shape[0], -1) # shape (B, N * C)
        recon = recon.view(functions.shape) # shape (B, N, C)
        return recon
