# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

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
    def estimate_captured_energy(
        self,
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C)
        *args,
        **kwargs,
    ) -> torch.Tensor: # (B, )
        """Estimate the energy captured by the dictionary.

        Computes an estimate of ``E_k[|<f, phi_k>|^2] = sum_k p(k) * |<f, phi_k>|^2``
        for each function ``f`` in the batch, where ``p(k)`` is the PMF of the
        atom sampling distribution and inner products are taken with respect to
        the measure described by ``logabsdet``.

        Args:
            coords: Quadrature points, shape ``(N, d)``.
            logabsdet: Log absolute value of the measure density / Jacobian at
                each point, shape ``(N,)``.
            values: Function values at the quadrature points, shape
                ``(B, N, C)``.
            *args: Additional positional arguments forwarded to the subclass.
            **kwargs: Additional keyword arguments forwarded to the subclass.

        Returns:
            Per-function energy estimate, shape ``(B,)``.
        """
        raise NotImplementedError("Subclasses must implement this method if needed")

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
