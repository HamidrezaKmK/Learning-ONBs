
import torch
from torch import nn
from typing import Literal

from .base import NeuralIsometry
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.dictionaries import InfiDictionary

class HouseholderIsometry(NeuralIsometry):

    def __init__(
        self,
        domain_dim: int,
        n_reflections: int,
        initial_diffeomorphism: Diffeomorphism | None = None
    ):
        super().__init__(initial_diffeomorphism=initial_diffeomorphism)

        self.n_reflections = n_reflections
        # create n neural fields
        for i in range(n_reflections):
            # TODO: maybe use Neural Field here?
            setattr(
                self,
                f"reflection_{i}",
                nn.Sequential(
                    nn.Linear(domain_dim, 2 * domain_dim),
                    nn.ReLU(),
                    nn.Linear(2 * domain_dim, 4 * domain_dim),
                    nn.ReLU(),
                    nn.Linear(4 * domain_dim, 1),
                )
            )
    

    def _action(
        self,
        coords: torch.Tensor, # (N, D)
        field_values: torch.Tensor, # (B, N)
        order: bool,
    ):
        idx_range = list(range(self.n_reflections))
        if not order:
            idx_range = list(reversed(idx_range))
        for i in idx_range:
            reflection = self._modules[f"reflection_{i}"]
            v = reflection(coords).squeeze() # (N,)
            v = v / torch.norm(v)
            inner_products = torch.einsum("bn,n->b", field_values, v) # (B, )
            interim = torch.einsum("b,n->bn", inner_products, v)
            field_values = field_values - 2 *  interim # (B, N)
        return field_values
    
    def pullback(
        self,
        coords: torch.Tensor, # (N, D)
        field_values: torch.Tensor, # (B, N)
    ):
        return self._action(coords, field_values, order=True)
        
    def pushforward(
        self,
        coords: torch.Tensor, # (N, D)
        field_values: torch.Tensor, # (B, N)
    ):
        return self._action(coords, field_values, order=False)
    
    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
    ):
        field_values = initial_dictionary.get_atom(coords, atom_indices).to(device) # (A, N)
        if mode == 'pullback':
            return self.pullback(
                coords,
                field_values,
            )
        else:
            return self.pushforward(
                coords,
                field_values,
            )
    