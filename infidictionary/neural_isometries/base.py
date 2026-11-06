import torch

from abc import ABC, abstractmethod

class NeuralIsometry(ABC, torch.nn.Module):

    """
    Implements a general abstract Neural Isometry that maps arbitrary
    functions to other functions while preserving inner products.

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

    def shuffle_model_state(self, **_state_kwargs):
        """Refresh any per-call state. Stateless isometries ignore the kwargs."""
        return self

class IdentityIsometry(NeuralIsometry):
    """The identity map, used as the no-isometry baseline. Ignores any transform kwargs."""

    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
        **_isometry_kwargs,
    ):
        return src_coords, src_logabsdet, src_field

    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
        **_isometry_kwargs,
    ):
        return tgt_coords, tgt_logabsdet, tgt_field
