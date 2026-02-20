from typing import Literal, Callable, Dict, Any
import torch

from .base import NeuralIsometry
from infidictionary.utils import TimeEvolvingField, norm2, pairwise_inner_product

class EulerianIsometry(NeuralIsometry):

    # TODO: implement forward mode backpropagation to avoid memory blowup
    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
    ):
        super().__init__()

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
        )
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.register_buffer("tspan", torch.tensor([]))

    def _householder_step(
        self,
        t: float,
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C) evaluated at coords
    ):
        device = coords.device
        dtype = coords.dtype
        N = values.shape[1]
        t1_batch = torch.ones(N, device=device, dtype=dtype) * t
        v = self.function_field(t1_batch, coords)  # (N, C)
        v = v / torch.sqrt(norm2(v, logabsdet)).unsqueeze(-1) # (N, C)
        inner_products = pairwise_inner_product(values, v, logabsdet) # (B, )
        interim = torch.einsum("b,nc->bnc", inner_products, v) # (B, N, C)
        values = values - 2 *  interim # (B, N, C)

        return values
        
    def _run_euler(
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        f: torch.Tensor, # (B, N, C)
        tspan: torch.Tensor, # (T,)
    ):
        for t0, t1 in zip(tspan[:-1], tspan[1:]):
            f = self._householder_step(t1, coords, logabsdet, f)
            f = self._householder_step(t0, coords, logabsdet, f)
        return f
    
    def shuffle_model_state(self, num_steps: int):
        if self.training:
            tspan = torch.rand(num_steps)
            # register as buffer
            self.tspan = torch.sort(tspan)[0]
        else:
            self.tspan = torch.linspace(0, 1, num_steps)
    
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        # sample random tspan for each forward pass
        tspan = self.tspan * (end_time - start_time) + start_time
        tgt_field = self._run_euler(
            src_coords,
            src_logabsdet,
            src_field,
            tspan=tspan,
        )
        return src_coords, src_logabsdet, tgt_field
    
    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        tspan = self.tspan.flip(0) * (end_time - start_time) + start_time
        src_field = self._run_euler(
            tgt_coords,
            tgt_logabsdet,
            tgt_field,
            tspan=tspan,
        )
        return tgt_coords, tgt_logabsdet, src_field # do not touch the volume terms
