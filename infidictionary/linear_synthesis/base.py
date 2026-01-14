
import torch
from torch import nn

class OrthogonalSynthesis(nn.Module):
    def __init__(
        self,
        n_atoms: int,
    ):
        super().__init__()
        self.n_atoms = n_atoms

    def forward(
        self, 
        initial_atoms: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the synthesized signal of size (B, N)"""
        raise NotImplementedError("OrthogonalSynthesis is not implemented yet.")

    def inverse(
        self,
        synthesized_atoms: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ):
        """Returns the initial atoms from the synthesized signal of size (B, N)"""
        raise NotImplementedError("OrthogonalSynthesis is not implemented yet.")
