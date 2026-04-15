from typing import Literal, Callable, Dict, Any
import torch
from torch.utils.checkpoint import checkpoint
import math

import wandb
from .base import NeuralIsometry
from infidictionary.networks import TimeEvolvingField
from infidictionary.utils import norm2, pairwise_inner_product, parallel_inner_product

class EulerianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        acceleration: float,
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
        self.acceleration = acceleration
        self._num_steps = 0
        self.register_buffer("tspan", torch.tensor([]), persistent=False)

    def _cayley_step(
        self,
        t0: float,
        t1: float,
        coords: torch.Tensor, # (N, d)
        logabsdet: torch.Tensor, # (N, )
        values: torch.Tensor, # (B, N, C) evaluated at coords
    ):
        device = coords.device
        dtype = coords.dtype
        N = values.shape[1]
        t_mid = (t0 + t1) / 2
        delta_t = t1 - t0
        t_mid_batch = torch.ones(N, device=device, dtype=dtype) * t_mid
        alpha_raw, beta_raw = self.function_field(t_mid_batch, coords.detach()) # (N, C), (N, C)
        
        alpha_alpha_raw = norm2(alpha_raw, logabsdet).clamp(min=1e-8) # scalar
        beta_beta_raw = norm2(beta_raw, logabsdet).clamp(min=1e-8) # scalar

        scale = math.sqrt(abs(delta_t) * self.acceleration)
        alpha_hat = alpha_raw * (scale / torch.sqrt(alpha_alpha_raw))
        beta_hat = beta_raw * (scale / torch.sqrt(beta_beta_raw))
        # Negate alpha to invert the skew operator when running backwards
        if delta_t < 0:
            alpha_hat = -alpha_hat
        alpha_alpha = norm2(alpha_hat, logabsdet).clamp(min=1e-8)
        beta_beta = norm2(beta_hat, logabsdet).clamp(min=1e-8)
        alpha_beta = pairwise_inner_product(alpha_hat, beta_hat, logabsdet)

        # NOTE: visualize cosine similarities
        # values_stopped = values.detach()              # (B, N, C)
        # values_values = parallel_inner_product(values_stopped, values_stopped, logabsdet) # (B, )
        # alpha_alpha_raw = norm2(alpha_raw, logabsdet).clamp(min=1e-8) # (1, )
        # beta_beta_raw = norm2(beta_raw, logabsdet).clamp(min=1e-8) # (1, )
        # values_alpha = pairwise_inner_product(values_stopped, alpha_raw, logabsdet) # (B, )
        # values_beta = pairwise_inner_product(values_stopped, beta_raw, logabsdet) # (B, )
        # cos_theta_alpha = values_alpha / torch.sqrt(values_values * alpha_alpha_raw).clamp(min=1e-8) # (B, )
        # cos_theta_beta = values_beta / torch.sqrt(values_values * beta_beta_raw).clamp(min=1e-8) # (B, )
        # cos_theta_alpha = torch.round(cos_theta_alpha, decimals=2)
        # cos_theta_beta = torch.round(cos_theta_beta, decimals=2)
        # print(f"t={t_mid:.4f},\n\tcos_theta_alpha={cos_theta_alpha.detach().tolist()},\n\tcos_theta_beta={cos_theta_beta.detach().tolist()}")


        #     cos_theta_alpha = values_alpha / torch.sqrt(values_values * alpha_alpha_raw).clamp(min=1e-8) # (B, )
        #     cos_theta_beta = values_beta / torch.sqrt(values_values * beta_beta_raw).clamp(min=1e-8) # (B, )
        # NOTE: prox operator-like
        # if self.training:
            
        #     # Straight-through gradient trick: forward values are unchanged, but
        #     # an auxiliary gradient signal is injected that pushes alpha_hat and
        #     # beta_hat to be similar to the current (partially-transformed) atoms.
        #     # values is detached so gradients only flow through alpha_hat/beta_hat.
        #     values_stopped = values.detach()              # (B, N, C)
        #     values_values = parallel_inner_product(values_stopped, values_stopped, logabsdet) # (B, )
        #     alpha_alpha_raw = norm2(alpha_raw, logabsdet).clamp(min=1e-8) # (1, )
        #     beta_beta_raw = norm2(beta_raw, logabsdet).clamp(min=1e-8) # (1, )
        #     values_alpha = pairwise_inner_product(values_stopped, alpha_raw, logabsdet) # (B, )
        #     values_beta = pairwise_inner_product(values_stopped, beta_raw, logabsdet) # (B, )

        #     cos_theta_alpha = values_alpha / torch.sqrt(values_values * alpha_alpha_raw).clamp(min=1e-8) # (B, )
        #     cos_theta_beta = values_beta / torch.sqrt(values_values * beta_beta_raw).clamp(min=1e-8) # (B, )

        #     L_alpha = -(cos_theta_alpha ** 2).mean() # encourage alpha_hat to be aligned with values
        #     L_beta = -(cos_theta_beta ** 2).mean() # encourage beta_hat to be aligned with values
        #     L_alpha_beta = (alpha_beta * delta_t) ** 2 # encourage alpha and beta to be orthogonal (so that the Cayley transform is well-conditioned)

        #     ((L_alpha + L_beta + L_alpha_beta) / num_steps).backward(retain_graph=True) # backprop through the function field, but not through the Cayley transform itself

        #     wandb.log({"train/L_alpha": L_alpha.item()}) # , step=epoch_i)
        #     wandb.log({"train/L_beta": L_beta.item()}) # , step=epoch_i)
        #     wandb.log({"train/L_alpha_beta": L_alpha_beta.item()}) # , step=epoch_i)


        # compute M = (I - 0.5 * V^* U)^{-1} via Woodbury
        a = 1 - 0.5 * alpha_beta
        b = -0.5 * beta_beta
        c = 0.5 * alpha_alpha
        d = 1 + 0.5 * alpha_beta
        inv_det = 1 / (a * d - b * c)
        M_00 = d * inv_det
        M_01 = -b * inv_det
        M_10 = -c * inv_det
        M_11 = a * inv_det

        c_alpha = pairwise_inner_product(values, alpha_hat, logabsdet) # (B, )
        c_beta = pairwise_inner_product(values, beta_hat, logabsdet) # (B, )
        y = values + 0.5 * (c_beta[:, None, None] * alpha_hat.unsqueeze(0) - c_alpha[:, None, None] * beta_hat.unsqueeze(0)) # (B, N, C)

        c_tilde_alpha = pairwise_inner_product(y, alpha_hat, logabsdet) # (B, )
        c_tilde_beta = pairwise_inner_product(y, beta_hat, logabsdet) # (B, )
        w_alpha = M_00 * c_tilde_beta - M_01 * c_tilde_alpha # (B, )
        w_beta = M_10 * c_tilde_beta - M_11 * c_tilde_alpha # (B, )

        ret = y + 0.5 * (w_alpha[:, None, None] * alpha_hat.unsqueeze(0) + w_beta[:, None, None] * beta_hat.unsqueeze(0))

        return ret
    
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
                def _step(f, _t0=t0_v, _t1=t1_v):
                    return self._cayley_step(_t0, _t1, coords, logabsdet, f)
                f = checkpoint(_step, f, use_reentrant=False)
            else:
                f = self._cayley_step(t0_v, t1_v, coords, logabsdet, f)
        return f

    def shuffle_model_state(self, num_steps: int | None = None):
        if num_steps is not None:
            self._num_steps = num_steps
        if self._num_steps == 0:
            raise ValueError("Run shuffle_model_state at least once with num_steps larger than 0 before this!")
        # TODO: fix later on!
        # if self.training:
        #     tspan = torch.rand(self._num_steps)
        #     self.tspan = torch.sort(tspan)[0]
        # else:
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
