import torch
from torch import nn

from .base import OrthogonalSynthesis

class IdentitySynthesis(OrthogonalSynthesis):
    
    def pullback(self, initial_atoms: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
        return initial_atoms

    def pushforward(self, synthesized_atoms: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
        return synthesized_atoms
