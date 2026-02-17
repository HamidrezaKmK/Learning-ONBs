import torch

from .base import NeuralIsometry
from infidictionary.diffeomorphisms.base import Diffeomorphism

class LagrangianIsometry(NeuralIsometry):
    """
    Neural isometry that changes the domain of the functions via a diffeomorphism.
    """

    def __init__(
        self,
        diffeomorphism: Diffeomorphism,
    ):
        super().__init__()
        self.diffeomorphism = diffeomorphism
    
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, c),
        *args,
        **kwargs,
    ):
        tgt_coords, log_abs_det_jacobian = self.diffeomorphism(src_coords, *args, **kwargs)
        tgt_field = src_field  # since it's an isometry, the function values don't change
        return tgt_coords, log_abs_det_jacobian + src_logabsdet, tgt_field * torch.exp(-0.5 * log_abs_det_jacobian)[None, :, None]

    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, c),
        *args,
        **kwargs,
    ):
        src_coords, log_abs_det_jacobian = self.diffeomorphism.inverse(tgt_coords, *args, **kwargs)
        return src_coords, log_abs_det_jacobian + tgt_logabsdet, tgt_field * torch.exp(-0.5 * log_abs_det_jacobian)[None, :, None]
