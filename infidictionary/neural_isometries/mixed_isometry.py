import torch
from typing import Literal

from abc import ABC, abstractmethod
from infidictionary.domain_samplers import DomainSampler
from .base import NeuralIsometry

class GeneralizedIsometry(NeuralIsometry):

    def __init__(
        self,
        first_change_of_domain: NeuralIsometry | None,
        ode_isometry: NeuralIsometry,
        last_change_of_domain: NeuralIsometry | None,
    ):
        super().__init__()
        self.first_change_of_domain = first_change_of_domain
        self.ode_isometry = ode_isometry
        self.last_change_of_domain = last_change_of_domain
        
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
        start_time: float,
        end_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.first_change_of_domain is not None:
            src_coords, src_logabsdet, src_field = self.first_change_of_domain.pushforward(
                src_coords, 
                src_logabsdet, 
                src_field,
            )
        src_coords, src_logabsdet, src_field = self.ode_isometry.pushforward(
            src_coords, 
            src_logabsdet, 
            src_field,
            start_time,
            end_time,
        )
        if self.last_change_of_domain is not None:
            src_coords, src_logabsdet, src_field = self.last_change_of_domain.pushforward(
                src_coords, 
                src_logabsdet, 
                src_field,
            )
        return src_coords, src_logabsdet, src_field

    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
        start_time: float,
        end_time: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the coordinates and values of the pullback operation
        Due to the isometry property, the pullback should be the inverse of the pushforward.
        """
        if self.last_change_of_domain is not None:
            tgt_coords, tgt_logabsdet, tgt_field = self.last_change_of_domain.pullback(
                tgt_coords, 
                tgt_logabsdet, 
                tgt_field,
            )
        tgt_coords, tgt_logabsdet, tgt_field = self.ode_isometry.pullback(
            tgt_coords, 
            tgt_logabsdet, 
            tgt_field,
            start_time=start_time,
            end_time=end_time,
        )
        if self.first_change_of_domain is not None:
            tgt_coords, tgt_logabsdet, tgt_field = self.first_change_of_domain.pullback(
                tgt_coords, 
                tgt_logabsdet, 
                tgt_field,
            )
        return tgt_coords, tgt_logabsdet, tgt_field

    def shuffle_model_state(self, *args, **kwargs):
        if self.first_change_of_domain is not None:
            self.first_change_of_domain.shuffle_model_state()
        self.ode_isometry.shuffle_model_state(*args, **kwargs)
        if self.last_change_of_domain is not None:
            self.last_change_of_domain.shuffle_model_state()
