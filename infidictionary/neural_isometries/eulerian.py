from typing import Literal, Callable, Dict, Any
import torch
from torch.utils.checkpoint import checkpoint

from .base import NeuralIsometry
from infidictionary.utils import TimeEvolvingField, norm2, pairwise_inner_product

class EulerianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
        )
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.gradient_checkpointing = gradient_checkpointing
        self._num_steps = 0
        self.register_buffer("tspan", torch.tensor([]), persistent=False)

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
        v = self.function_field(t1_batch, coords.detach())  # (N, C)
        v = v / torch.sqrt(norm2(v, logabsdet).clamp(min=1e-8)).unsqueeze(-1) # (N, C)
        inner_products = pairwise_inner_product(values, v, logabsdet) # (B, )
        interim = torch.einsum("b,nc->bnc", inner_products, v) # (B, N, C)
        values = values - 2 * interim # (B, N, C)

        return values

    def _run_euler(
        self,
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        f: torch.Tensor, # (B, N, C)
        tspan: torch.Tensor, # (T,)
    ):
        use_ckpt = self.gradient_checkpointing and self.training
        for t0_v, t1_v in zip(tspan[:-1].tolist(), tspan[1:].tolist()):
            if use_ckpt:
                # Capture t values as defaults so the closure doesn't share a mutable loop variable.
                # checkpoint stores only the input f and recomputes the two Householder steps
                # during backward — O(1) intermediate activations instead of O(num_steps).
                def _two_steps(f, _t0=t0_v, _t1=t1_v):
                    f = self._householder_step(_t1, coords, logabsdet, f)
                    f = self._householder_step(_t0, coords, logabsdet, f)
                    return f
                f = checkpoint(_two_steps, f, use_reentrant=False)
            else:
                f = self._householder_step(t1_v, coords, logabsdet, f)
                f = self._householder_step(t0_v, coords, logabsdet, f)
        return f

    def shuffle_model_state(self, num_steps: int | None = None):
        if num_steps is not None:
            self._num_steps = num_steps
        if self._num_steps == 0:
            raise ValueError("Run shuffle_model_state at least once with num_steps larger than 0 before this!")
        if self.training:
            tspan = torch.rand(self._num_steps)
            self.tspan = torch.sort(tspan)[0]
        else:
            self.tspan = torch.linspace(0, 1, self._num_steps)

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
