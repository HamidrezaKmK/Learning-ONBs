from typing import Literal, Callable
import math
import torch
from torch import nn
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms.base import Diffeomorphism
from .base import NeuralIsometry
from infidictionary.utils import NeuralField, TimeEmbedding, SinusoidalTimeEmbedding
from .householder import TimeEvolvingField

class EulerianDynamics(torch.nn.Module):
    def __init__(
        self, 
        field_partial: Callable, 
        coord_dims: int, 
        value_dims: int,
        time_embedding: TimeEmbedding | None = None,
    ):
        super().__init__()
        self.time_embedding = time_embedding or SinusoidalTimeEmbedding()
        self.coord_dims = coord_dims
        self.value_dims = value_dims
        self.alpha_field = field_partial(input_dim=self.time_embedding.out_dim + coord_dims, output_dim=1)
        self.beta_field = field_partial(input_dim=self.time_embedding.out_dim + coord_dims, output_dim=1)
    
    def field_values(self, t: torch.Tensor, coords: torch.Tensor, kind=Literal['alpha', 'beta']) -> torch.Tensor:
        # t: (B, )
        # x: (B, d)
        t_batch = torch.ones_like(coords[:, 0]) * t  # (N, )
        t_emb = self.time_embedding(t_batch.unsqueeze(-1))  # (B, time_embedding_dim)
        inp = torch.cat([t_emb, coords], dim=-1)  # (B, time_embedding_dim + d)
        ret = self.alpha_field(inp) if kind == 'alpha' else self.beta_field(inp)  # (B, 1)
        return ret.squeeze(-1)  # (B, )
    
    # TODO: can I use a package that can do adjoint method for this?
    def forward(self, timesteps, coords, values):
        
        delta_t = timesteps[1:] - timesteps[:-1]  # (T-1, )
        timesteps_reversed = timesteps[:-1].flip(0)  # (T-1, ) # TODO: not sure about reversal
        for t, dt in zip(timesteps_reversed, delta_t):
            
            alpha = self.field_values(t, coords, kind='alpha')  # (N,)
            beta = self.field_values(t, coords, kind='beta')    # (N,)

            xb = (beta[None, :] * values).mean(dim=-1)  # (B, )
            xa = (alpha[None, :] * values).mean(dim=-1)  # (B, )
            
            # do the first half step of Cayley transform (I + 0.5 dt \alpha \otimes \beta - 0.5 dt \beta \otimes \alpha)
            values = values + 0.5 * (xb[:, None] * alpha - xa[:, None] * beta) * dt  # (B, value_dims)
            

            xb = (beta[None, :] * values).mean(dim=-1)  # (B, )
            xa = (alpha[None, :] * values).mean(dim=-1)  # (B, )

            aa = (alpha * alpha).mean() # (1, )
            ab = (alpha * beta).mean()  # (1, )
            bb = (beta * beta).mean()    # (1, )

            A1 = torch.stack([
                torch.stack([1 + 0.5 * dt * ab,   -0.5 * dt * aa]),
                torch.stack([0.5 * dt * bb,        1 - 0.5 * dt * ab]),
            ], dim=0)  # (2,2) keeps grad

            B1 = torch.stack([xa, xb], dim=0)  # (2, B)
            sol1 = torch.linalg.lstsq(A1, B1).solution   # (2, B)
            a, b = sol1[0, :], sol1[1, :]  # (B, ), (B, )

            # Solve for s,r (per function)
            A2 = torch.stack([
                torch.stack([aa, ab]),
                torch.stack([ab, bb]),
            ], dim=0)  # (2,2) keeps grad

            B2 = torch.stack([a - xa, b - xb], dim=0)                                           # (2,B)
            sol2 = torch.linalg.lstsq(A2, B2).solution   # (2,B)
            s, r = sol2[0], sol2[1]                                                             # (B,)

            # apply correction y = x + s alpha + r beta
            values = values + s[:, None] * alpha[None, :] + r[:, None] * beta[None, :]

        return coords, values
    
class LowRankEulerianIsometry(NeuralIsometry):
    
    def __init__(
        self, 
        eulerian_dynamics: EulerianDynamics,
        n_steps: int,
        initial_diffeomorphism: Diffeomorphism | None = None,
    ):
        super().__init__(initial_diffeomorphism=initial_diffeomorphism)
        self.eulerian_dynamics = eulerian_dynamics
        self.n_steps = n_steps

    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
    ):
        grid_values = initial_dictionary.get_atom(coords, atom_indices).to(device) # (A, N)

        if mode == 'pullback':
            _, all_deformed_vals_pullback = self.eulerian_dynamics(
                timesteps=torch.linspace(0.0, 1.0, self.n_steps, device=device),
                coords=coords,
                values=grid_values,
            )  # (A, N)
            return all_deformed_vals_pullback
        else:
            _, all_deformed_vals_pushforward = self.eulerian_dynamics(
                timesteps=torch.linspace(1.0, 0.0, self.n_steps, device=device),
                coords=coords,
                values=grid_values,
            )  # (A, N)
            return all_deformed_vals_pushforward
