import torch
from typing import Literal

from abc import ABC, abstractmethod
from infidictionary.domain_samplers import DomainSampler

class NeuralIsometry(ABC, torch.nn.Module):

    """
    Implements a general abstract Neural Isometry that maps arbitrary
    functions to other functions while preserving inner products.

    The general abstraction takes in domain sampler object which essentially
    samples coordinates on the domain w.r.t. some pre-defined probability measure.

    The pushforward and pullback methods take in coordinates, along their field values
    at those coordinates, then the function returns the transformed coordinates, and
    transformed function values at those coordinates.
    """
    def __init__(
        self,
    ):
        super().__init__()
        
    @abstractmethod
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and values of the pushforward operation
        """
        pass

    @abstractmethod
    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and values of the pullback operation
        Due to the isometry property, the pullback should be the inverse of the pushforward.
        """
        pass

    def train(self, mode: bool = True):
        result = super().train(mode)
        self.shuffle_model_state()
        return result

    def shuffle_model_state(self):
        return self

class IdentityIsometry(NeuralIsometry):
    """
    A toy neural isometry that does nothing, but is still an isometry. Useful for debugging and testing.
    """

    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
        start_time: float | None = None, # optional time argument for time-dependent isometries
        end_time: float | None = None, # optional time argument for time-dependent isometries
    ):
        return src_coords, src_logabsdet, src_field

    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d) 
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
        start_time: float | None = None, # optional time argument for time-dependent isometries
        end_time: float | None = None, # optional time argument for time-dependent isometries
    ):
        return tgt_coords, tgt_logabsdet, tgt_field
