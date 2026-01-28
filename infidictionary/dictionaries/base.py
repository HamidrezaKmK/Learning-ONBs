# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import torch

class InfiDictionary(ABC):

    def __init__(
        self,
        atom_indices: list[int] | None = None,
        num_atoms: int | None = None,
        numerical_gram_n_samples: int | None = None,
        is_orthonormal: bool = False,
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
        
        if self.atom_indices is not None:
            self.atom_indices_tensor = torch.tensor(self.atom_indices, dtype=torch.long)
        else:
            self.atom_indices_tensor = None

        self.is_orthonormal = is_orthonormal
        self.numerical_gram_n_samples = numerical_gram_n_samples or 100_000

    @abstractmethod
    def sample_from_domain(self, N: int):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def _get_atoms(self, coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement this method if needed")

    def get_atom(self, coords: torch.Tensor, idx: int | torch.Tensor):
        if self.atom_indices is None:
            real_idx = torch.tensor([idx]).cpu() if isinstance(idx, int) else idx.cpu()
            atoms = self._get_atoms(coords, real_idx)
            return atoms.squeeze(0)
        elif isinstance(idx, torch.Tensor):
            real_idx = self.atom_indices_tensor[idx.cpu()]
            return self._get_atoms(coords, real_idx)
        else:
            real_idx = torch.tensor([self.atom_indices[idx]])
            return self._get_atoms(coords, real_idx).squeeze(0)
    
    def check_gram_computed(self, atom_indices: torch.Tensor):
        if not hasattr(self, '_gram_matrix_inv') or not hasattr(self, '_saved_atom_indices'):
            return False
        return torch.equal(self._saved_atom_indices, atom_indices.cpu(), exact=True)
    
    def gram_solve(self, atom_indices: torch.Tensor, inner_products: torch.Tensor):
        # TODO: perhaps the type of atom_indices should be a list
        """
        Inputs is a sequence of inner products of the functions and the atoms specified by atom_indices.
        Should in principle do a gram matrix solve where G_ij = <atom_{atom_indices[i]}, atom_{atom_indices[j]}>
        and return the coefficients.

        In many cases, the gram matrix is identity or diagonal, so this can be to just return inner_products.
        """
        if self.is_orthonormal:
            return inner_products
        else:
            device = inner_products.device
            if self.num_atoms is None:
                raise ValueError("Cannot compute Gram matrix for infinite basis.")
            if not self.check_gram_computed(atom_indices):
                N = self.numerical_gram_n_samples
                coords = self.sample_from_domain(N).cpu()
                all_f = self.get_atom(coords, atom_indices).cpu()  # shape (A, N)
                gram_matrix = all_f @ all_f.T / N  # shape (A, A)
                self._gram_matrix_inv = torch.linalg.pinv(gram_matrix)
                self._saved_atom_indices = atom_indices.cpu().long()
            gram_matrix_inv = self._gram_matrix_inv.to(device)
            coeffs = torch.einsum("ij,jb->ib", gram_matrix_inv, inner_products)  # shape (A, B)
            return coeffs

class MixedDictionary(InfiDictionary):
    """
    The mixture of multiple dictionaries by just combining their atoms
    """
    # TODO: if the dictionary is finite store the gram matrix for everything
    def __init__(self, dictionaries: List[InfiDictionary] | Dict[str, InfiDictionary]):
        if isinstance(dictionaries, list):
            self.dictionaries: List[InfiDictionary] = dictionaries
        else:
            self.dictionaries: List[InfiDictionary] = list(dictionaries.values())

        for dictionary in self.dictionaries:
            if dictionary.atom_indices is None:
                raise ValueError("All atoms in MixedDictionary must have finite atom_indices.")
        
        self._map = torch.zeros((sum(len(d.atom_indices) for d in self.dictionaries), 2), dtype=torch.long)
        idx = 0
        for dictionary_idx, dictionary in enumerate(self.dictionaries):
            for b_idx in dictionary.atom_indices:
                self._map[idx][0] = dictionary_idx
                self._map[idx][1] = b_idx
                idx += 1

        super().__init__(num_atoms=idx, is_orthonormal=False)
        
    def _get_atoms(self, coords: torch.Tensor, idx: torch.Tensor):
        dictionary_indices = self._map[idx][:, 0]
        atom_indices = self._map[idx][:, 1]
        d_ids, inverse_indices = torch.unique(dictionary_indices, return_inverse=True)
        results = torch.zeros((len(idx), coords.shape[0]), device=coords.device)
        for i, d_id in enumerate(d_ids):
            mask = (inverse_indices == i)
            b_idxs = atom_indices[mask]
            results[mask] = self.dictionaries[d_id.item()]._get_atoms(coords, b_idxs)
        return results    

    def sample_from_domain(self, N: int):
        if len(self.dictionaries) == 0:
            raise ValueError("MixedDictionary has no atoms to sample from.")
        return self.dictionaries[0].sample_from_domain(N)
