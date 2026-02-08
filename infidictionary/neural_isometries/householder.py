from typing import Literal, Callable
import torch
from torch import nn

from .base import NeuralIsometry
from infidictionary.diffeomorphisms.base import Diffeomorphism
from infidictionary.dictionaries import InfiDictionary
from infidictionary.utils import NeuralField, TimeEmbedding, SinusoidalTimeEmbedding
from torchdiffeq import odeint, odeint_adjoint

   
class HouseholderIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
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
                    nn.Linear(coords_dim, 2 * coords_dim),
                    nn.ReLU(),
                    nn.Linear(2 * coords_dim, 4 * coords_dim),
                    nn.ReLU(),
                    nn.Linear(4 * coords_dim, 1),
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


class TimeEvolvingField(NeuralField):
    def __init__(
        self,
        base_field_partial: Callable,
        coords_dim: int = 2,
        output_dim: int = 1,
    ):
        super().__init__(input_dim=coords_dim, output_dim=output_dim)
        self.time_embedding = SinusoidalTimeEmbedding() # TODO: fix this!
        self.time_evolving_field = base_field_partial(
            input_dim=self.time_embedding.out_dim + coords_dim,
            output_dim=output_dim,
        )
        self.coords_dim = coords_dim
        self.output_dim = output_dim

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # t: (B, )
        # x: (B, d)
        t_emb = self.time_embedding(t.unsqueeze(-1))  # (B, time_embedding_dim)
        inp = torch.cat([t_emb, x], dim=-1)  # (B, time_embedding_dim + d)
        return self.time_evolving_field(inp)  # (B, output_dim)
 

class InfinitesimalGivensIsometry(NeuralIsometry):

    class GivensDynamics(nn.Module):
        def __init__(
            self,
            givens_generator: 'InfinitesimalGivensIsometry',
        ):
            super().__init__()
            self.givens_generator = givens_generator

        def forward(
            self,
            t: torch.Tensor, # (,)
            y: torch.Tensor, # (B, N * d + N)
        ):
            
            
            def _inner_forward(self, coords: torch.Tensor, f: torch.Tensor, create_graph: bool):
                t_batch = torch.ones(coords.shape[0], device=coords.device, dtype=coords.dtype) * t
                t_batch = t_batch.requires_grad_(True)
                # evaluate the function field at the coordinates
                v = self.householder_generator.function_field(t_batch, coords).squeeze()  # (N,)
                v = v / torch.norm(v)
                # take the gradient of v w.r.t. time to get the derivative of the function field
                v_dot = torch.autograd.grad(v, t_batch, grad_outputs=torch.ones_like(v), create_graph=True)[0]  # (N,) 
                fv = torch.einsum("bn,n->b", f, v)  # (B, )
                fv_dot = torch.einsum("bn,n->b", f, v_dot)  # (B, )
                
                d_f = -2 * fv_dot.unsqueeze(-1) * v - 2 * fv.unsqueeze(-1) * v_dot  # (B, N)
                return d_f

            batch_size = y.shape[0]
            N_times_d_plus_one = y.shape[1]
            d = self.householder_generator.coords_dim
            N = N_times_d_plus_one // (d + 1)
            coords = y[0, :N * d].reshape(N, d) # (N, d)
            f = y[:, N * d:].reshape(batch_size, N) # (B, N)

            if torch.is_grad_enabled():
                d_f = _inner_forward(self, coords, f, create_graph=True)
            else:
                with torch.enable_grad():
                    d_f = _inner_forward(self, coords, f, create_graph=False)

            # coordinates will be the same as before so the derivatives are zero
            d_coords = torch.zeros_like(y[:, :N * d])  # (B, N * d)

            return torch.cat([d_coords, d_f], dim=-1)  # (B, N * d + N)

    def __init__(
        self,
        coords_dim: int,
        vector_field_partial: Callable,
        use_adjoint: bool = False,
        method: str = 'dopri5',
        rtol: float = 1e-5,
        atol: float = 1e-6,
        start_time: float = 0.0,
        end_time: float = 1.0,
        initial_diffeomorphism: Diffeomorphism | None = None
    ):
        super().__init__(
            initial_diffeomorphism=initial_diffeomorphism
        )
        self.function_field = vector_field_partial(
            coords_dim=coords_dim,
            output_dim=1,
        )
        self.coords_dim = coords_dim
        self.use_adjoint = use_adjoint
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.start_time = start_time
        self.end_time = end_time

    def _euler_householder(
        self,
        timesteps: torch.Tensor, # (T, )
        coords: torch.Tensor, # (N, d)
        values: torch.Tensor, # (B, N)
    ):
        for t0, t1 in zip(timesteps[:-1], timesteps[1:]):
            t1_batch = torch.ones(coords.shape[0], device=coords.device, dtype=coords.dtype) * t1
            v1 = self.function_field(t1_batch, coords).squeeze()  # (N,)
            v1 = v1 / torch.norm(v1)
            inner_products = torch.einsum("bn,n->b", values, v1) # (B, )
            interim = torch.einsum("b,n->bn", inner_products, v1)
            values = values - 2 *  interim # (B, N)

            t0_batch = torch.ones(coords.shape[0], device=coords.device, dtype=coords.dtype) * t0
            v0 = self.function_field(t0_batch, coords).squeeze()  # (N,)
            v0 = v0 / torch.norm(v0)
            inner_products = torch.einsum("bn,n->b", values, v0) # (B, )
            interim = torch.einsum("b,n->bn", inner_products, v0)
            values = values - 2 *  interim # (B, N)
            
        return values
    
        
    def _run_ode( # TODO: check and see if there's any bug in the solver
        self, 
        coords: torch.Tensor, # (N, d)
        f: torch.Tensor, # (B, N)
        forward: bool,
        num_steps: int = 200,
    ):
        # initial augmented state
        # repeat coords to make it (B, N, d) and flatten to (B, N * d)
        batch_size = f.shape[0]
        N = coords.shape[0]
        d = coords.shape[1]
        coords_flat = coords.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size, N * d) # (B, N * d)
        y0 = torch.cat([coords_flat, f], dim=-1) # (B, N * d + N)

        if forward:
            tspan = torch.tensor([self.start_time, self.end_time], device=coords.device, dtype=coords.dtype)
        else:
            tspan = torch.tensor([self.end_time, self.start_time], device=coords.device, dtype=coords.dtype)

        if self.method == 'euler_householder':
            # discretize tspan
            timesteps = torch.linspace(tspan[0], tspan[1], steps=num_steps, device=coords.device, dtype=coords.dtype)
            return self._euler_householder(
                timesteps=timesteps,
                coords=coords,
                values=f,
            )

        func = self.GivensDynamics(self)

        if self.use_adjoint:
            yt = odeint_adjoint(func, y0, tspan, method=self.method, rtol=self.rtol, atol=self.atol, adjoint_params=tuple(self.parameters()))
        else:
            yt = odeint(func, y0, tspan, method=self.method, rtol=self.rtol, atol=self.atol)

        fT = yt[-1, :, N * d:] # (B, N)
        return fT
    
    def forward(
        self,
        coords: torch.Tensor, # (N, d)
        f: torch.Tensor, # (B, N)
        *args,
        **kwargs,
    ):
        return self._run_ode(
            coords,
            f,
            forward=True,
            *args,
            **kwargs,
        ) 
    
    def inverse(
        self,
        coords: torch.Tensor, # (N, d)
        f: torch.Tensor, # (B, N)
        *args,
        **kwargs,
    ):
        return self._run_ode(
            coords,
            f,
            forward=False,
            *args,
            **kwargs,
        ) 
    
    def transform(
        self,
        initial_dictionary: InfiDictionary,
        atom_indices: torch.Tensor,
        coords: torch.Tensor, # (N, d)
        device: torch.device,
        mode: Literal['pullback', 'pushforward'],
        *args,
        **kwargs,
    ):
        field_values = initial_dictionary.get_atom(coords, atom_indices).to(device) # (A, N)
        if mode == 'pullback':
            return self.forward(
                coords,
                field_values,
                *args,
                **kwargs,
            )
        else:
            return self.inverse(
                coords,
                field_values,
                *args,
                **kwargs,
            )
