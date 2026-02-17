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
    