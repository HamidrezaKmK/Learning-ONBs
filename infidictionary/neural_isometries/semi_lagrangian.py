from typing import Literal, Callable, Dict, Any
import torch
from torch import nn

from .eulerian import EulerianIsometry
from .lagrangian import LagrangianIsometry
from .base import NeuralIsometry
from infidictionary.networks import SinusoidalTimeEmbedding, TimeEvolvingField
from infidictionary.diffeomorphisms import Diffeomorphism

class SemiLagrangianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        diffeomorphism: Diffeomorphism,
        gating_function: Callable[[torch.Tensor], torch.Tensor] | None = None, 
        time_tolerance: float = 1e-3,
    ):
        super().__init__()

        self.eulerian = EulerianIsometry(
            coords_dim=coords_dim,
            channels_dim=channels_dim,
            scalar_field_partial=scalar_field_partial,
        )
        self.lagrangian = LagrangianIsometry(diffeomorphism)
        # model a neural network that takes in time and outputs a scalar logit
        if gating_function is None:
            self.register_module(
                'gating_function', 
                nn.Sequential(
                    SinusoidalTimeEmbedding(),
                    nn.Linear(16, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2),
                )
            )
        else:
            self.gating_function = gating_function

        self.register_buffer("gating_draw_times", torch.tensor([]))
        self.register_buffer("gumbel_samples", torch.tensor([]))
        self.time_tolerance = time_tolerance
    
    def shuffle_model_state(self, num_steps: int):
        self.eulerian.shuffle_model_state(num_steps)
        self.gating_draw_times = torch.tensor([])
        self.gumbel_samples = torch.tensor([])

    def _compute_gating(self, t: torch.Tensor):
        logits = self.gating_function(t)  # (B, 2)
        # get gumbel noise for both classes
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))  # (B, 2)

        if len(self.gating_draw_times) == 0:
            self.gating_draw_times = t
            self.gumbel_samples = gumbel_noise
        else:
            # find the times that are already drawn
            time_distances = torch.abs(self.gating_draw_times[:, None] - t[None, :])  # (num_draws, B)
            closest_draw_indices = torch.argmin(time_distances, dim=0)  # (B,)
            closest_draw_distances = time_distances[closest_draw_indices, torch.arange(t.shape[0])]  # (B,)
            old_gumbel_samples = self.gumbel_samples[closest_draw_indices]  # (B,)
            new_draw_mask = closest_draw_distances > self.time_tolerance # (B, ) 
            gumbel_noise = torch.where(
                new_draw_mask.unsqueeze(-1).repeat(1, 2), # (B, 2)
                gumbel_noise,
                old_gumbel_samples,
            )
            self.gating_draw_times = torch.cat([self.gating_draw_times, t[new_draw_mask]])
            self.gumbel_samples = torch.cat([self.gumbel_samples, gumbel_noise[new_draw_mask]])
        # perform softmax and argmax
        probs_soft = torch.softmax(logits + gumbel_noise, dim=-1)  # (B, 2)
        probs_hard = torch.argmax(logits + gumbel_noise, dim=-1)  # (B,)
        # make it (B, 2) one-hot
        gating = torch.nn.functional.one_hot(probs_hard, num_classes=2).float()  # (B, 2)
        return gating - probs_soft.detach() + probs_soft  # (B, 2) with straight-through estimator

    def _run_solver(
        self, 
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        f: torch.Tensor, # (B, N, C)
        tspan: torch.Tensor, # (T,)
        gating: torch.Tensor, # (T, 2) binary tensor indicating whether to use Eulerian or Lagrangian step at each time step
    ):
        for t0, t1, p in zip(tspan[:-1], tspan[1:], gating):

            f = self.eulerian._cayley_step(t0, t1, coords, logabsdet, f)
            if t1 > t0:
                coords_lagrangian, logabsdet_lagrangian, f_lagrangian = self.lagrangian.pushforward(coords, logabsdet, f, start_time=t0.item(), end_time=t1.item())
            else:
                coords_lagrangian, logabsdet_lagrangian, f_lagrangian = self.lagrangian.pullback(coords, logabsdet, f, start_time=t1.item(), end_time=t0.item())
            coords = p[0] * coords + p[1] * coords_lagrangian
            logabsdet = p[0] * logabsdet + p[1] * logabsdet_lagrangian
            f = p[0] * f_eulerian + p[1] * f_lagrangian
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
        tspan_mid = tspan_mid.to(src_coords.device)
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
        tspan_mid = tspan_mid.to(tgt_coords.device)
        gating = self._compute_gating(tspan_mid)
        return self._run_solver(
            tgt_coords,
            tgt_logabsdet,
            tgt_field,
            tspan=tspan,
            gating=gating,
        )