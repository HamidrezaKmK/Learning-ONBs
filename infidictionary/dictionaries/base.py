from abc import ABC, abstractmethod
from typing import Iterator

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import orthogonal

from infidictionary.utils import haar_orthogonal, pairwise_inner_product


def orthogonal_synthesis(size: int, learn: bool) -> nn.Module | None:
    """Square orthogonal synthesis matrix: Haar-random and trainable, or a frozen identity."""
    if size == 0:
        return None
    layer = nn.Linear(size, size, bias=False)
    with torch.no_grad():
        init = haar_orthogonal(size, layer.weight.dtype) if learn else torch.eye(size, dtype=layer.weight.dtype)
        layer.weight.copy_(init)
    layer = orthogonal(layer)
    if not learn:
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
    return layer


class InfiDictionary(nn.Module, ABC):
    """Abstract base class for dictionaries over a continuous domain.

    A dictionary is a collection of *atoms* — scalar- or vector-valued
    functions on a continuous domain — equipped with a probability mass
    function (PMF) over its index set.  The index set may be either

    * **infinite** (e.g. the real Fourier basis on ``[0,1]^d``, with a
      summable power-law prior over frequencies), or
    * **finite** (e.g. a set of fixed resolution hat functions or Voronoi cell 
      step functions).

    Subclasses implement specific atom families and decide how the index set is
    laid out, how atoms are evaluated, how indices are sampled from the PMF, and
    what probability each index carries.  Everything beyond these three core
    operations is dictionary-specific and lives on the subclass.

    Atoms are evaluated at a finite set of *coordinates* (quadrature points) and
    are identified by integer *indices* whose meaning is dictionary-specific.

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
        synthesis: bool = True,
    ) -> torch.Tensor: # (A, N, C)
        """Evaluate dictionary atoms at the given coordinates.

        Args:
            coords: Quadrature / sample points, shape ``(N, d)``.
            idx: Integer indices selecting which atoms to evaluate,
                shape ``(A, ...)``.  The inner dimensions are
                dictionary-specific (e.g. ``(A, d)`` for multi-index atoms).
            synthesis: Whether to apply the dictionary's optional synthesis
                transform. ``False`` requests its raw/base atoms. Dictionaries
                without a synthesis transform return the same atoms either way.

        Returns:
            Atom values at each coordinate, shape ``(A, N, C)``; entry
            ``(i, j, c)`` is the ``j``-th evaluation of the ``i``-th atom on
            channel ``c``.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_indices(
        self,
        num_samples: int,
        with_replacement: bool,
    ) -> torch.Tensor:
        """Draw atom indices according to the dictionary's PMF.

        Args:
            num_samples: Number of atom indices to draw.
            with_replacement: If ``True``, draws are i.i.d. from the PMF and
                may repeat.  If ``False``, the returned indices are distinct.

        Returns:
            Sampled indices, shape ``(num_samples, ...)`` matching the layout
            of :meth:`get_atoms`'s ``idx`` argument.
        """
        raise NotImplementedError

    @abstractmethod
    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Return the prior probability ``p(k)`` for each atom index.

        Args:
            idx: Integer indices, shape ``(A, ...)`` matching :meth:`get_atoms`.

        Returns:
            Probability tensor of shape ``(A,)``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return every atom index with PMF at least ``tail_probability``."""
        raise NotImplementedError

    def monte_carlo_captured_energy(
        self,
        coords: torch.Tensor,
        logabsdet: torch.Tensor,
        values: torch.Tensor,
        num_tail_samples: int,
        tail_probability: float = 1e-4,
    ) -> torch.Tensor:
        """Estimate captured energy using an exact PMF stratum and a sampled tail."""
        idx_exact = self.get_high_probability_indices(tail_probability).to(coords.device)
        atoms_exact = self.get_atoms(coords, idx_exact)
        coeffs_exact = pairwise_inner_product(values, atoms_exact, logabsdet)
        pmfs_exact = self.get_index_pmfs(idx_exact).to(coords.device)
        energy_exact = (coeffs_exact.square() * pmfs_exact[None, :]).sum(dim=-1)

        if num_tail_samples <= 0:
            return energy_exact

        idx_all = self.sample_indices(num_tail_samples, with_replacement=True).to(coords.device)
        in_exact = (idx_all[:, None, :] == idx_exact[None, :, :]).all(dim=-1).any(dim=-1)
        idx_tail = idx_all[~in_exact]
        if idx_tail.numel() == 0:
            return energy_exact

        idx_tail, counts = torch.unique(idx_tail, return_counts=True, dim=0)
        atoms_tail = self.get_atoms(coords, idx_tail)
        coeffs_tail = pairwise_inner_product(values, atoms_tail, logabsdet)
        energy_tail = (
            coeffs_tail.square() * counts[None, :].to(coeffs_tail.dtype)
        ).sum(dim=-1) / num_tail_samples
        return energy_exact + energy_tail

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Trainable parameters owned by the dictionary itself.

        A dictionary with trainable overrides this to yield those 
        parameters, so the training loop can hand them to the optimizer 
        alongside the isometry's — letting the initial dictionary itself 
        be learned jointly. This is mostly useful for linear synthesis
        style mixing that can be learned, e.g. instead of using a fixed
        Fourier basis, allow for a quadratic mixing of the Fourier bases.
        """
        return (
            parameter
            for parameter in super().parameters(recurse=recurse)
            if parameter.requires_grad
        )
