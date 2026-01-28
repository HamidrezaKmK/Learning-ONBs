
import torch
from torch import nn

class OrthogonalSynthesis(nn.Module):
    def __init__(
        self,
        n_atoms: int,
    ):
        super().__init__()
        self.n_atoms = n_atoms

    def pullback(
        self,
        synthesized_atoms: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ):
        """Returns the initial atoms from the synthesized signal of size (B, N)"""
        raise NotImplementedError("OrthogonalSynthesis is not implemented yet.")

    def pushforward(
        self,
        initial_dictionary: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ):
        """Returns the synthesized atoms from the initial signal of size (B, N)"""
        raise NotImplementedError("OrthogonalSynthesis is not implemented yet.")
    
    def forward(
        self, 
        initial_dictionary: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the synthesized signal of size (B, N)"""
        return self.pullback(initial_dictionary, atom_indices)
    