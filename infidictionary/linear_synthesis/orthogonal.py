import torch
from torch import nn

from .base import OrthogonalSynthesis

class HouseholderSynthesis(OrthogonalSynthesis):

    def __init__(self, n_atoms: int, n_householder: int, eps: float = 1e-12):
        super().__init__(n_atoms=n_atoms)
        self.V = nn.Parameter(torch.randn(n_householder, n_atoms))
        self.eps = eps

    def _transform(self, initial_dictionary: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """
        Apply a sequence of Householder reflections:
            f <- (I - 2 v v^T) f
        where v is a unit vector in R^{n_atoms}.
        initial_dictionary: (n_atoms, N)
        returns:       (n_atoms, N)
        """
        f = initial_dictionary
        for i in range(V.shape[0]):
            v = V[i]  # (n_atoms,)
            v = v / (v.norm(p=2) + self.eps)  # (n_atoms,)
            inner = v.unsqueeze(0) @ f # (1, N)
            f = f - 2.0 * v.unsqueeze(1) @ inner # (n_atoms, N)

        return f
    
    def pushforward(self, initial_dictionary: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:    
        V = self.V.to(dtype=initial_dictionary.dtype, device=initial_dictionary.device)[:, atom_indices]
        return self._transform(initial_dictionary, V)

    def pullback(self, synthesized_atoms: torch.Tensor, atom_indices: torch.Tensor) -> torch.Tensor:
        V = self.V.to(dtype=synthesized_atoms.dtype, device=synthesized_atoms.device)[:, atom_indices]
        # invert the order of the rows in V
        V = torch.flip(V, dims=[0])
        return self._transform(synthesized_atoms, V)

