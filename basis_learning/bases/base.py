# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import torch

class BaseFunction(ABC):

    def __init__(
        self,
        basis_indices: list[int] | None = None,
        num_basis_elements: int | None = None,
    ):
        if basis_indices is not None:
            self.basis_indices = basis_indices
            self.num_basis_elements = len(basis_indices)
        elif num_basis_elements is not None:
            self.num_basis_elements = num_basis_elements
            self.basis_indices = list(range(num_basis_elements))
        else:
            self.basis_indices = None
            self.num_basis_elements = None
        
    @abstractmethod
    def __call__(self, coords: torch.Tensor, *args, **kwds):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def sample_from_domain(self, N: int):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _get(self, coords: torch.Tensor, idx: int):
        raise NotImplementedError("Subclasses must implement this method if needed")

    def get(self, coords: torch.Tensor, idx: int):
        if self.basis_indices is None:
            real_idx = idx
        else:
            real_idx = self.basis_indices[idx]

        return self._get(coords, real_idx)
    
    def compute_gram_matrix(self, n_domain_samples: int, device: torch.device):
        if self.num_basis_elements is None:
            raise ValueError("Cannot compute Gram matrix for infinite basis.")
        N = n_domain_samples
        coords = self.sample_from_domain(N).to(device)
        n_funcs = len(self.basis_indices)
        gram_matrix = torch.zeros((n_funcs, n_funcs), device=device)
        for i in range(n_funcs):
            for j in range(i+1):
                fi = self.get(coords, self.basis_indices[i])
                fj = self.get(coords, self.basis_indices[j])
                integrand: torch.Tensor = fi * fj
                gram_matrix[i, j] = integrand.mean()
                gram_matrix[j, i] = gram_matrix[i, j]
        self._gram_matrix_inv = torch.linalg.pinv(gram_matrix)
    
    @property
    def gram_matrix_inv(self):
        if not hasattr(self, "_gram_matrix_inv"):
            raise ValueError("Gram matrix inverse has not been computed yet. Call compute_gram_matrix first.")
        return self._gram_matrix_inv

class MixtureBasis(BaseFunction):
    """
    The mixture of multiple bases
    """
    def __init__(self, bases: List[BaseFunction] | Dict[str, BaseFunction]):
        if isinstance(bases, list):
            self.bases = bases
        else:
            self.bases = list(bases.values())

        for base in self.bases:
            if base.basis_indices is None:
                raise ValueError("All bases in MixtureBasis must have finite basis_indices.")
        
        self._map = {}
        idx = 0
        for base_idx, base in enumerate(self.bases):
            for b_idx in base.basis_indices:
                self._map[idx] = (base_idx, b_idx)
                idx += 1

        super().__init__(num_basis_elements=idx)
        
    def _get(self, coords: torch.Tensor, idx: int):
        if idx not in self._map:
            raise ValueError(f"Index {idx} not found in MixtureBasis.")
        base_idx, b_idx = self._map[idx]
        return self.bases[base_idx]._get(coords, b_idx)
    
    def __call__(self, coords, *args, **kwds):
        raise NotImplementedError("MixtureBasis does not implement __call__; use get() instead.")
    
    def sample_from_domain(self, N: int):
        if len(self.bases) == 0:
            raise ValueError("MixtureBasis has no bases to sample from.")
        return self.bases[0].sample_from_domain(N)
