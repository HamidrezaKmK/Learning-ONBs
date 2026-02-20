# import abstractmethod and abstract class
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

import torch


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
    