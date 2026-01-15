
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
        initial_atoms: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ):
        """Returns the synthesized atoms from the initial signal of size (B, N)"""
        raise NotImplementedError("OrthogonalSynthesis is not implemented yet.")
    
    def forward(
        self, 
        initial_atoms: torch.Tensor, # (B, N)
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the synthesized signal of size (B, N)"""
        return self.pullback(initial_atoms, atom_indices)
    
    def truncated_orthonormal( # return Q_L^T
        self, 
        n_truncation: int,
        atom_indices: torch.Tensor,
    ) -> torch.Tensor:
        # create an (n_truncation, n_atoms) matrix with identity in the first n_truncation rows
        inp = torch.cat([
            torch.eye(n_truncation), 
            torch.zeros((n_truncation, self.n_atoms - n_truncation))], 
            dim=1,
        )
        return self.forward(inp, atom_indices) # (B, N) # TODO: this is probably wrong!
        