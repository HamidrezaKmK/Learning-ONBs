from typing import Literal, Callable, Dict, Any
import torch
from torch import nn

from .eulerian import EulerianIsometry
from .lagrangian import LagrangianIsometry
from .base import NeuralIsometry
from infidictionary.utils import TimeEvolvingField, SinusoidalTimeEmbedding
from infidictionary.diffeomorphisms import Diffeomorphism

class SemiLagrangianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        diffeomorphism: Diffeomorphism,
        gating_function: Callable[[torch.Tensor], torch.Tensor], 
        time_tolerance: float = 1e-3,
        # a function that takes in time and outputs a scalar in [0, 1] that gates between Eulerian and Lagrangian steps. If None, use a learnable gating function.
        # TODO: find a way to train gating and backpropagate through it using policy gradient tricks
    ):
        super().__init__()

        self.eulerian = EulerianIsometry(
            coords_dim=coords_dim,
            channels_dim=channels_dim,
            scalar_field_partial=scalar_field_partial,
        )
        self.lagrangian = LagrangianIsometry(diffeomorphism)
        # model a neural network that takes in time and outputs a scalar in [0, 1] that gates between Eulerian and Lagrangian steps
        self.gating_function = gating_function
        self.register_buffer("gating_draw_times", torch.tensor([]))
        self.register_buffer("gating_draws", torch.tensor([]))
        self.time_tolerance = time_tolerance
    
    def reshuffle_time_discretization(self, num_steps: int):
        self.eulerian.reshuffle_time_discretization(num_steps)
        self.gating_draw_times = torch.tensor([])
        self.gating_draws = torch.tensor([])

    def _compute_gating(self, t: torch.Tensor):
        probabilities = self.gating_function(t)  # (B,)
        if self.training:
            ret = torch.bernoulli(probabilities).squeeze(-1)  # (B,)
        else:
            ret = (probabilities > 0.5).float().squeeze(-1)  # (B,)
        
        if len(self.gating_draw_times) == 0:
            self.gating_draw_times = t
            self.gating_draws = ret
            return ret
        
        # find the times that are already drawn
        time_distances = torch.abs(self.gating_draw_times[:, None] - t[None, :])  # (num_draws, B)
        closest_draw_indices = torch.argmin(time_distances, dim=0)  # (B,)
        closest_draw_distances = time_distances[closest_draw_indices, torch.arange(t.shape[0])]  # (B,)
        old_gating_draws = self.gating_draws[closest_draw_indices]  # (B,)

        new_draw_mask = closest_draw_distances > self.time_tolerance # (B, )
        
        ret = torch.where(
            new_draw_mask,
            ret,
            old_gating_draws,
        )
        self.gating_draw_times = torch.cat([self.gating_draw_times, t[new_draw_mask]])
        self.gating_draws = torch.cat([self.gating_draws, ret[new_draw_mask]])

        return ret

    def _run_solver(
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        f: torch.Tensor, # (B, N, C)
        tspan: torch.Tensor, # (T,)
        gating: torch.Tensor, # (T,) binary tensor indicating whether to use Eulerian or Lagrangian step at each time step
    ):
        for t0, t1, p in zip(tspan[:-1], tspan[1:], gating):
            if p > 0.5:
                f = self.eulerian._householder_step(t1, coords, logabsdet, f)
                f = self.eulerian._householder_step(t0, coords, logabsdet, f)
            else:
                if t1 > t0:
                    coords, logabsdet, f = self.lagrangian.pushforward(coords, logabsdet, f, start_time=t0.item(), end_time=t1.item())
                else:
                    coords, logabsdet, f = self.lagrangian.pullback(coords, logabsdet, f, start_time=t1.item(), end_time=t0.item())

        return coords, logabsdet, f
    
    
    def pushforward(
        self,
        src_coords: torch.Tensor, # (N, d)
        src_logabsdet: torch.Tensor, # (N, )
        src_field: torch.Tensor, # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        # sample random tspan for each forward pass
        tspan = self.eulerian.tspan * (end_time - start_time) + start_time
        tspan_mid = (tspan[:-1] + tspan[1:]) / 2
        gating = self._compute_gating(tspan_mid)
        return self._run_solver(
            src_coords,
            src_logabsdet,
            src_field,
            tspan=tspan,
            gating=gating,
        )
    
    def pullback(
        self,
        tgt_coords: torch.Tensor, # (N, d)
        tgt_logabsdet: torch.Tensor, # (N, )
        tgt_field: torch.Tensor, # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        tspan = (self.eulerian.tspan * (end_time - start_time) + start_time).flip(0)
        tspan_mid = (tspan[:-1] + tspan[1:]) / 2
        gating = self._compute_gating(tspan_mid)
        return self._run_solver(
            tgt_coords,
            tgt_logabsdet,
            tgt_field,
            tspan=tspan,
            gating=gating,
        )