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
        sub_size: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        gradient_checkpointing: bool = False,
        use_spatial_clustering: bool = False,
    ):
        super().__init__()

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
        )
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.use_spatial_clustering = use_spatial_clustering
        self.acceleration = acceleration
        self.sub_size = sub_size
        self._num_steps = 0
        self.register_buffer("tspan", torch.tensor([]), persistent=False)

    def _cayley_step(
        self,
        t0: float,
        t1: float,
        alpha_raw: torch.Tensor, # (K, S, C)
        beta_raw: torch.Tensor,  # (K, S, C)
        logabsdet: torch.Tensor, # (K, S)
        values: torch.Tensor,    # (B, K, S, C)
    ):
        alpha_alpha_raw = parallel_inner_product(alpha_raw, alpha_raw, logabsdet).clamp(min=1e-8) # (K, )
        beta_beta_raw   = parallel_inner_product(beta_raw,  beta_raw,  logabsdet).clamp(min=1e-8) # (K, )
        # ensure it is at least dimension 1
        alpha_alpha_raw = torch.atleast_1d(alpha_alpha_raw)
        beta_beta_raw   = torch.atleast_1d(beta_beta_raw)


        delta_t = t1 - t0
        scale = math.sqrt(abs(delta_t) * self.acceleration)
        alpha_hat = alpha_raw * (scale / torch.sqrt(alpha_alpha_raw))[:, None, None]
        beta_hat  = beta_raw  * (scale / torch.sqrt(beta_beta_raw))[:, None, None]
        
        if delta_t < 0:
            alpha_hat = -alpha_hat


        alpha_alpha = parallel_inner_product(alpha_hat, alpha_hat, logabsdet).clamp(min=1e-8) # (K, )
        beta_beta   = parallel_inner_product(beta_hat,  beta_hat,  logabsdet).clamp(min=1e-8) # (K, )
        alpha_beta  = parallel_inner_product(alpha_hat, beta_hat, logabsdet) # (K, )
        alpha_alpha = torch.atleast_1d(alpha_alpha)
        beta_beta   = torch.atleast_1d(beta_beta)
        alpha_beta  = torch.atleast_1d(alpha_beta)

        # M = (I - ½ V* U)^{-1} via Woodbury (2×2 inverse)
        a = 1 - 0.5 * alpha_beta;  b = -0.5 * beta_beta
        c = 0.5 * alpha_alpha;     d =  1 + 0.5 * alpha_beta
        inv_det = 1 / (a * d - b * c)
        M_00, M_01 = d * inv_det, -b * inv_det
        M_10, M_11 = -c * inv_det,  a * inv_det

        exp_logabsdet = torch.exp(logabsdet)
        c_alpha = torch.einsum("bksc, ksc, ks -> bk", values, alpha_hat, exp_logabsdet) / self.sub_size # (B, K)
        c_beta  = torch.einsum("bksc, ksc, ks -> bk", values, beta_hat,  exp_logabsdet) / self.sub_size # (B, K)

        y = values + 0.5 * (c_beta[:, :, None, None] * alpha_hat.unsqueeze(0) - c_alpha[:, :, None, None] * beta_hat.unsqueeze(0))

        c_tilde_alpha = torch.einsum("bksc, ksc, ks -> bk", y, alpha_hat, exp_logabsdet) / self.sub_size # (B, K)
        c_tilde_beta  = torch.einsum("bksc, ksc, ks -> bk", y, beta_hat,  exp_logabsdet) / self.sub_size # (B, K)

        w_alpha = M_00 * c_tilde_beta - M_01 * c_tilde_alpha # (B, K)
        w_beta  = M_10 * c_tilde_beta - M_11 * c_tilde_alpha # (B, K)

        ret = y + 0.5 * (w_alpha[:, :, None, None] * alpha_hat.unsqueeze(0) + w_beta[:, :, None, None] * beta_hat.unsqueeze(0))
        return ret
    
    def _run_euler_sub(
        self,
        logabsdet: torch.Tensor,     # (K, S,)
        f: torch.Tensor,             # (B, K, S, C)
        tspan: torch.Tensor,         # (T,)
        alpha_all: torch.Tensor,     # (T-1, K, S, C)
        beta_all: torch.Tensor,      # (T-1, K, S, C)
    ):
        use_ckpt = self.gradient_checkpointing and self.training
        idx = 0
        for t0_v, t1_v in zip(tspan[:-1].tolist(), tspan[1:].tolist()):
            if use_ckpt:
                # Capture the current index by default argument so the checkpoint doesn't use a stale reference
                def _step(v, _t0=t0_v, _t1=t1_v, _idx=idx):
                    return self._cayley_step(_t0, _t1, alpha_all[_idx], beta_all[_idx], logabsdet, v)
                f = checkpoint(_step, f, use_reentrant=False)
            else:
                f = self._cayley_step(t0_v, t1_v, alpha_all[idx], beta_all[idx], logabsdet, f)
            
            idx += 1
            
        return f

    def _get_balanced_spatial_clusters(self, coords: torch.Tensor, S: int) -> torch.Tensor:
        """
        Recursively bisects spatial coordinates to form spatially localized 
        clusters of exactly size S. Returns the permutation indices.
        """
        with torch.no_grad():
            N = coords.shape[0]
            indices = torch.arange(N, device=coords.device)
            
            def recurse(idx):
                n = len(idx)
                if n <= S:
                    return idx
                
                # Get coordinates for the current subset
                sub_coords = coords[idx]
                
                # Find the spatial dimension with the largest variance
                variances = torch.var(sub_coords, dim=0)
                split_dim = torch.argmax(variances)
                
                # Sort points along the axis of maximum variance
                sort_order = torch.argsort(sub_coords[:, split_dim])
                idx_sorted = idx[sort_order]
                
                # Split exactly at a multiple of S to guarantee shape compatibility
                k = n // S
                half_k = k // 2
                split_point = half_k * S
                
                left_idx = recurse(idx_sorted[:split_point])
                right_idx = recurse(idx_sorted[split_point:])
                
                return torch.cat([left_idx, right_idx])
            
            return recurse(indices)
        
    def _run_euler(
        self,
        coords: torch.Tensor,    # (N, d)
        logabsdet: torch.Tensor, # (N,)
        f: torch.Tensor,         # (B, N, C)
        tspan: torch.Tensor,     # (T,)
    ):
        B, N, C = f.shape
        S = self.sub_size
        
        # Strict divisibility check
        if N % S != 0:
            raise ValueError(f"Spatial dimension N ({N}) must be exactly divisible by sub_size ({S}).")
            
        K = N // S  # K is the number of strata/chunks

        t_mid = (tspan[:-1] + tspan[1:]) / 2  # (T-1,)
        t_repeated = t_mid.unsqueeze(1).unsqueeze(2).repeat(1, N, 1)  # (T-1, N, 1)
        coords_repeated = coords.unsqueeze(0).repeat(t_mid.shape[0], 1, 1)  # (T-1, N, d)

        # PRECOMPUTE: Evaluate the function field
        alpha_all, beta_all = self.function_field(
            t_repeated.reshape(-1, 1).to(coords.device), # (T-1)*N, 1 
            coords_repeated.reshape(-1, self.coords_dim)
        )
        alpha_all = alpha_all.reshape(t_mid.shape[0], N, self.channels_dim)  # (T-1, N, C)
        beta_all = beta_all.reshape(t_mid.shape[0], N, self.channels_dim)  # (T-1, N, C)

        # 1. Stratification Strategy Selection
        if self.use_spatial_clustering:
            # Spatially-Correlated Strict Clustering (Balanced KD-Tree)
            coord_shuffle = self._get_balanced_spatial_clusters(coords, S)
        else:
            # Fully random stratified subsampling
            coord_shuffle = torch.randperm(N, device=coords.device)

        # Apply the chosen grouped indices
        l_shuf = logabsdet[coord_shuffle]
        f_shuf = f[:, coord_shuffle]
        alpha_shuf = alpha_all[:, coord_shuffle] 
        beta_shuf = beta_all[:, coord_shuffle]
        
        # 2. Reshape: Push Stratum (K) into Batch (B)
        f_batched = f_shuf.reshape(B, K, S, C)
        alpha_batched = alpha_shuf.reshape(t_mid.shape[0], K, S, C)  # (T-1, K, S, C)
        beta_batched = beta_shuf.reshape(t_mid.shape[0], K, S, C)    # (T-1, K, S, C)
        l_batched = l_shuf.reshape(K, S)  # (K, S)

        # 3. Solve ODE
        f_batched_updated = self._run_euler_sub(
            logabsdet=l_batched, 
            f=f_batched, 
            tspan=tspan, 
            alpha_all=alpha_batched, 
            beta_all=beta_batched
        )  # (B, K, S, C)
        
        # 4. Restore Shapes
        f_shuf_updated = f_batched_updated.reshape(B, N, C)
            
        # 5. Unshuffle OUT-OF-PLACE to preserve Autograd
        inverse_shuffle = torch.argsort(coord_shuffle)
        f_out = f_shuf_updated[:, inverse_shuffle]
        
        return f_out

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
