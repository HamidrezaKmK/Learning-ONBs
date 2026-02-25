# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import torch

from infidictionary.utils import pairwise_inner_product

class InfiDictionary(ABC):
    
    @abstractmethod
    def get_atoms(
        self, 
        coords: torch.Tensor, # (N, d)
        idx: torch.Tensor, # (A, ...)
    ) -> torch.Tensor: # (A, N, C) 
        raise NotImplementedError("Subclasses must implement this method if needed")

    @abstractmethod
    def sample_indices(self, num_samples: int) -> torch.Tensor:
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
        raise NotImplementedError("Subclasses must implement this method if needed")
    
    @abstractmethod
    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
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
