import torch
from typing import Literal

from abc import ABC, abstractmethod
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms.base import Diffeomorphism 
from infidictionary.diffeomorphisms import IdentityFlow

class NeuralIsometry(ABC, torch.nn.Module):

    def __init__(
        self,
        initial_diffeomorphism: Diffeomorphism | None = None,
    ):
        super().__init__()
        self.initial_diffeomorphism = initial_diffeomorphism or IdentityFlow()
        
    @abstractmethod
    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
    ):
        raise NotImplementedError("Subclasses must implement this method")
    
    def inner_products(
        self,
        atom_indices: torch.Tensor, # (A, )
        coords: torch.Tensor, # (N, d)
        vals: torch.Tensor, # (B, N)
        initial_dictionary: InfiDictionary,
        device: torch.device,
        return_pullback: bool = False,
    ): # TODO: there are more elaborate ways of doing inner products with fourier and over time pullback pushforwards
        """
        Compute the inner products on the deformed dictionary atoms in atom_indices
        and the functions vals sampled at coords.

        Also return whether or not the same computation graph is used or not.
        """
        N = coords.shape[0]
        all_deformed_vals_pullback = self.transform(
            initial_dictionary=initial_dictionary,
            atom_indices=atom_indices,
            coords=coords,
            device=device,
            mode='pullback',
        )  # (A, N)
        inner_products = torch.einsum("an,bn->ab", all_deformed_vals_pullback, vals) / N # (A, B)
        if return_pullback:
            return inner_products, all_deformed_vals_pullback
        return inner_products

class IdentityIsometry(NeuralIsometry):
    """
    Identity neural isometry that does not change the dictionary atoms.
    """

    def __init__(self):
        super().__init__(initial_diffeomorphism=IdentityFlow())

    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
    ):
        return initial_dictionary.get_atom(coords, atom_indices)  # (A, N)  
