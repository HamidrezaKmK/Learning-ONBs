# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import torch

# TODO: vectorize the get_atom code to allow for more parallelization
# TODO: replace the gram matrix interfact with some like a Gram inverse action with an index
class InfiDictionary(ABC):

    def __init__(
        self,
        atom_indices: list[int] | None = None,
        num_atoms: int | None = None,
    ):
        if atom_indices is not None:
            self.atom_indices = atom_indices
            self.num_atoms = len(atom_indices)
        elif num_atoms is not None:
            self.num_atoms = num_atoms
            self.atom_indices = list(range(num_atoms))
        else:
            self.atom_indices = None
            self.num_atoms = None

    @abstractmethod
    def sample_from_domain(self, N: int):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _get_atom(self, coords: torch.Tensor, idx: int):
        raise NotImplementedError("Subclasses must implement this method if needed")

    def get_atom(self, coords: torch.Tensor, idx: int):
        if self.atom_indices is None:
            real_idx = idx
        else:
            real_idx = self.atom_indices[idx]

        return self._get_atom(coords, real_idx)
    
    def compute_gram_matrix(self, n_domain_samples: int, device: torch.device):
        if self.num_atoms is None:
            raise ValueError("Cannot compute Gram matrix for infinite basis.")
        N = n_domain_samples
        coords = self.sample_from_domain(N).to(device)
        n_funcs = len(self.atom_indices)
        gram_matrix = torch.zeros((n_funcs, n_funcs), device=device)
        for i in range(n_funcs):
            for j in range(i+1):
                fi = self.get_atom(coords, self.atom_indices[i])
                fj = self.get_atom(coords, self.atom_indices[j])
                integrand: torch.Tensor = fi * fj
                gram_matrix[i, j] = integrand.mean()
                gram_matrix[j, i] = gram_matrix[i, j]
        self._gram_matrix_inv = torch.linalg.pinv(gram_matrix)
    
    @property
    def gram_matrix_inv(self):
        if not hasattr(self, "_gram_matrix_inv"):
            raise ValueError("Gram matrix inverse has not been computed yet. Call compute_gram_matrix first.")
        return self._gram_matrix_inv

class MixedDictionary(InfiDictionary):
    """
    The mixture of multiple dictionaries by just combining their atoms
    """
    def __init__(self, atoms: List[InfiDictionary] | Dict[str, InfiDictionary]):
        if isinstance(atoms, list):
            self.atoms = atoms
        else:
            self.atoms = list(atoms.values())

        for dictionary in self.atoms:
            if dictionary.atom_indices is None:
                raise ValueError("All atoms in MixedDictionary must have finite atom_indices.")
        
        self._map = {}
        idx = 0
        for dictionary_idx, dictionary in enumerate(self.atoms):
            for b_idx in dictionary.atom_indices:
                self._map[idx] = (dictionary_idx, b_idx)
                idx += 1

        super().__init__(num_atoms=idx)
        
    def _get_atom(self, coords: torch.Tensor, idx: int):
        if idx not in self._map:
            raise ValueError(f"Index {idx} not found in MixedDictionary.")
        dictionary_idx, b_idx = self._map[idx]
        return self.atoms[dictionary_idx]._get_atom(coords, b_idx)

    def sample_from_domain(self, N: int):
        if len(self.atoms) == 0:
            raise ValueError("MixedDictionary has no atoms to sample from.")
        return self.atoms[0].sample_from_domain(N)
