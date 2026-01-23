from typing import Literal
import math
import torch
from torch import nn
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms.base import Diffeomorphism
from .base import NeuralIsometry

class HalfDensityIsometry(NeuralIsometry):
    
    def __init__(
        self, 
        diffeomorphism: Diffeomorphism,
        initial_diffeomorphism: Diffeomorphism | None = None,
    ):
        super().__init__(initial_diffeomorphism=initial_diffeomorphism)
        self.diffeomorphism = diffeomorphism

    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
    ):
        if mode == 'pullback':
            deformed_coords, logabsdets = self.diffeomorphism(coords)  # (N, d)
        else:
            deformed_coords, logabsdets = self.diffeomorphism.inverse(coords)
        
        transformation = initial_dictionary.get_atom(deformed_coords, atom_indices).to(device) # (A, N)

        return transformation * torch.exp(0.5 * logabsdets) # (A, N)
