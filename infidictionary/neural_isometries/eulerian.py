from typing import Callable, Dict, Any, Literal
import torch
from torch.utils.checkpoint import checkpoint
import math

from .base import NeuralIsometry
from infidictionary.networks import TimeEvolvingField
from infidictionary.utils import parallel_inner_product


class EulerianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        base_acceleration: float,
        sub_size: int,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        gradient_checkpointing: bool = False,
        clustering: Literal["random", "spatial"] = "random",
        multigrid_strategy: Literal["just_fine", "just_global", "multigrid"] = "just_fine",
    ):
        super().__init__()

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
        )
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.clustering = clustering
        self.multigrid_strategy = multigrid_strategy
        self.base_acceleration = base_acceleration
        self.sub_size = sub_size
        self._num_steps = 0
        self._model_state_seed: int = 0
        self.register_buffer("tspan", torch.tensor([]), persistent=False)

    # ── Cayley step ────────────────────────────────────────────────────────────
    #
    # One method serves all three granularities by varying (K, S):
    #
    #   fine:   alpha (K, S, C),  logabsdet (K, S)  — K independent local steps
    #   coarse: alpha (1, K, C),  logabsdet (1, K)  — one step on K cluster means
    #           (caller collapses S via mean and broadcasts correction back after)
    #   global: alpha (1, N, C),  logabsdet (1, N)  — one step on all N points

    def _cayley_step(
        self,
        t0: float,
        t1: float,
        alpha_raw: torch.Tensor,   # (K, S, C)
        beta_raw: torch.Tensor,    # (K, S, C)
        logabsdet: torch.Tensor,   # (K, S)
        values: torch.Tensor,      # (B, K, S, C)
        acceleration: float,
    ) -> torch.Tensor:             # (B, K, S, C)
        """Cayley isometry on K independent groups of S quadrature points.

        Normalises α̂, β̂ to L²-norm = scale = √(|Δt|·acceleration), then applies
        the rank-2 Cayley map f ↦ (I − K/2)⁻¹(I + K/2)f  with  K = α̂⊗β̂* − β̂⊗α̂*
        via the Woodbury 2×2 identity.  Exact L²-isometry for any K, S, C.
        Negating Δt (i.e. swapping t0, t1) produces the exact inverse.
        """
        delta_t = t1 - t0
        scale = math.sqrt(abs(delta_t) * acceleration)

        aa_raw = torch.atleast_1d(parallel_inner_product(alpha_raw, alpha_raw, logabsdet).clamp(min=1e-8))
        bb_raw = torch.atleast_1d(parallel_inner_product(beta_raw,  beta_raw,  logabsdet).clamp(min=1e-8))

        alpha_hat = alpha_raw * (scale / aa_raw.sqrt())[:, None, None]   # (K, S, C)
        beta_hat  = beta_raw  * (scale / bb_raw.sqrt())[:, None, None]   # (K, S, C)
        if delta_t < 0:
            alpha_hat = -alpha_hat

        aa = torch.atleast_1d(parallel_inner_product(alpha_hat, alpha_hat, logabsdet).clamp(min=1e-8))
        bb = torch.atleast_1d(parallel_inner_product(beta_hat,  beta_hat,  logabsdet).clamp(min=1e-8))
        ab = torch.atleast_1d(parallel_inner_product(alpha_hat, beta_hat,  logabsdet))

        # 2×2 Woodbury inverse M = (I − ½V*U)⁻¹, one per group k
        a = 1 - 0.5 * ab;  b = -0.5 * bb
        c = 0.5 * aa;       d =  1 + 0.5 * ab
        inv_det = 1.0 / (a * d - b * c)
        M_00, M_01 = d * inv_det, -b * inv_det
        M_10, M_11 = -c * inv_det,  a * inv_det

        S = values.shape[2]
        exp_lad = torch.exp(logabsdet)                                                      # (K, S)
        c_a = torch.einsum("bksc,ksc,ks->bk", values, alpha_hat, exp_lad) / S              # (B, K)
        c_b = torch.einsum("bksc,ksc,ks->bk", values, beta_hat,  exp_lad) / S              # (B, K)

        y = values + 0.5 * (c_b[:, :, None, None] * alpha_hat - c_a[:, :, None, None] * beta_hat)

        ct_a = torch.einsum("bksc,ksc,ks->bk", y, alpha_hat, exp_lad) / S                  # (B, K)
        ct_b = torch.einsum("bksc,ksc,ks->bk", y, beta_hat,  exp_lad) / S                  # (B, K)

        w_a = M_00 * ct_b - M_01 * ct_a                                                     # (B, K)
        w_b = M_10 * ct_b - M_11 * ct_a                                                     # (B, K)

        return y + 0.5 * (w_a[:, :, None, None] * alpha_hat + w_b[:, :, None, None] * beta_hat)

    # ── Inner Euler loop ───────────────────────────────────────────────────────

    def _run_euler_sub(
        self,
        logabsdet: torch.Tensor,   # (K, S)
        f: torch.Tensor,           # (B, K, S, C)
        tspan: torch.Tensor,       # (T,)
        alpha_all: torch.Tensor,   # (T-1, K, S, C)
        beta_all: torch.Tensor,    # (T-1, K, S, C)
    ):
        use_ckpt = self.gradient_checkpointing and self.training
        K, S = logabsdet.shape
        N = K * S

        need_coarse = (self.multigrid_strategy == "multigrid")
        if need_coarse:
            alpha_k_all = alpha_all.mean(dim=2)                                              # (T-1, K, C)
            beta_k_all  = beta_all.mean(dim=2)                                               # (T-1, K, C)
            l_k = torch.log(torch.exp(logabsdet).mean(dim=1).clamp(min=1e-8))                # (K,)

        for idx, (t0_v, t1_v) in enumerate(zip(tspan[:-1].tolist(), tspan[1:].tolist())):

            # ── Fine: K independent local Cayley steps ──────────────────────────
            def _fine(v, _t0=t0_v, _t1=t1_v, _idx=idx):
                return self._cayley_step(
                    _t0, _t1,
                    alpha_all[_idx], beta_all[_idx], logabsdet, v,
                    self.base_acceleration,
                )

            # ── Coarse: single Cayley step on K cluster means, correction ────────
            #    broadcast to all S fine points within each cluster
            def _coarse(v, _t0=t0_v, _t1=t1_v, _idx=idx, _S=S):
                B = v.shape[0]
                f_k = v.mean(dim=2)                                                          # (B, K, C)
                f_k_new = self._cayley_step(
                    _t0, _t1,
                    alpha_k_all[_idx].unsqueeze(0),                                          # (1, K, C)
                    beta_k_all[_idx].unsqueeze(0),                                           # (1, K, C)
                    l_k.unsqueeze(0),                                                        # (1, K)
                    f_k.unsqueeze(1),                                                        # (B, 1, K, C)
                    self.base_acceleration * _S,
                )                                                                             # (B, 1, K, C)
                return v + (f_k_new[:, 0] - f_k).unsqueeze(2)                               # (B, K, S, C)

            # ── Global: single Cayley step over all N points with full α(t,x) ──
            #    c_α = Σ_n exp(lad_n)·f_n·α(x_n)/N  — true L² inner product,
            #    mixing ALL Fourier modes simultaneously
            def _global(v, _t0=t0_v, _t1=t1_v, _idx=idx, _K=K, _S=S, _N=N):
                B = v.shape[0]
                return self._cayley_step(
                    _t0, _t1,
                    alpha_all[_idx].reshape(1, _N, -1),                                      # (1, N, C)
                    beta_all[_idx].reshape(1, _N, -1),                                       # (1, N, C)
                    logabsdet.reshape(1, _N),                                                 # (1, N)
                    v.reshape(B, 1, _N, -1),                                                  # (B, 1, N, C)
                    self.base_acceleration * _N,
                ).reshape(B, _K, _S, -1)

            # Palindrome fine→coarse→global→coarse→fine is self-inverse under Δt<0:
            # (F_fine∘F_coarse∘F_global∘F_coarse∘F_fine)⁻¹
            #   = F_fine⁻¹∘F_coarse⁻¹∘F_global⁻¹∘F_coarse⁻¹∘F_fine⁻¹
            # which is the same list with each step called with swapped t0,t1 (Δt<0).
            if self.multigrid_strategy == "just_fine":
                ordered = [_fine]
            elif self.multigrid_strategy == "just_global":
                ordered = [_global]
            elif self.multigrid_strategy == "multigrid":  # multigrid
                ordered = [_fine, _coarse, _global, _coarse, _fine]
            else:
                raise ValueError(f"Invalid multigrid_strategy: {self.multigrid_strategy}")
            
            for step_fn in ordered:
                if use_ckpt:
                    f = checkpoint(step_fn, f, use_reentrant=False)
                else:
                    f = step_fn(f)

        return f

    # ── Spatial clustering helper ──────────────────────────────────────────────

    def _get_balanced_spatial_clusters(self, coords: torch.Tensor, S: int) -> torch.Tensor:
        """Balanced axis-aligned KD-tree bisection; returns a permutation of [0, N).
        Deterministic given _model_state_seed; reshuffled only when shuffle_model_state() is called.
        """
        with torch.no_grad():
            N = coords.shape[0]
            d = coords.shape[1]
            indices = torch.arange(N, device=coords.device)
            g = torch.Generator()
            g.manual_seed(self._model_state_seed)

            def recurse(idx):
                n = len(idx)
                if n <= S:
                    return idx
                sub_coords = coords[idx]
                # direction = torch.zeros(d, device=coords.device)
                # direction[torch.randint(0, d, (1,), device=coords.device)] = 1
                direction = torch.randn(d, generator=g).to(coords.device)
                direction = direction.to(coords.dtype)
                sort_order = torch.argsort(sub_coords @ direction)
                idx_sorted = idx[sort_order]
                half_k = (n // S) // 2
                split  = half_k * S
                return torch.cat([recurse(idx_sorted[:split]), recurse(idx_sorted[split:])])

            return recurse(indices)

    # ── Top-level Euler integrator ─────────────────────────────────────────────

    def _run_euler(
        self,
        coords: torch.Tensor,    # (N, d)
        logabsdet: torch.Tensor, # (N,)
        f: torch.Tensor,         # (B, N, C)
        tspan: torch.Tensor,     # (T,)
    ):
        B, N, C = f.shape
        S = self.sub_size

        if N % S != 0:
            raise ValueError(f"N={N} must be divisible by sub_size={S}.")

        K = N // S

        t_mid       = (tspan[:-1] + tspan[1:]) / 2                                         # (T-1,)
        T1          = len(t_mid)
        t_flat      = t_mid.unsqueeze(1).expand(-1, N).reshape(-1).to(f.device)             # (T1·N,)
        coords_flat = coords.unsqueeze(0).expand(T1, -1, -1).reshape(-1, self.coords_dim)   # (T1·N, d)

        alpha_flat, beta_flat = self.function_field(t_flat, coords_flat)
        alpha_all = alpha_flat.reshape(T1, N, C)   # (T-1, N, C)
        beta_all  = beta_flat.reshape(T1, N, C)    # (T-1, N, C)

        if self.clustering == "spatial":
            shuffle = self._get_balanced_spatial_clusters(coords, S)
        else:  # "random"
            g = torch.Generator()
            g.manual_seed(self._model_state_seed)
            shuffle = torch.randperm(N, generator=g).to(f.device)

        f_shuf     = f[:, shuffle]
        l_shuf     = logabsdet[shuffle]
        alpha_shuf = alpha_all[:, shuffle]
        beta_shuf  = beta_all[:, shuffle]

        f_batched = self._run_euler_sub(
            l_shuf.reshape(K, S),
            f_shuf.reshape(B, K, S, C),
            tspan,
            alpha_shuf.reshape(T1, K, S, C),
            beta_shuf.reshape(T1, K, S, C),
        )

        inverse = torch.argsort(shuffle)
        return f_batched.reshape(B, N, C)[:, inverse]

    # ── Public interface ───────────────────────────────────────────────────────

    def shuffle_model_state(self, num_steps: int | None = None):
        if num_steps is not None:
            self._num_steps = num_steps
        if self._num_steps == 0:
            raise ValueError("Call shuffle_model_state with num_steps > 0 first.")
        self.tspan = torch.linspace(0, 1, self._num_steps)
        self._model_state_seed = int(torch.randint(0, 2**31, (1,)).item())

    def pushforward(
        self,
        src_coords: torch.Tensor,    # (N, d)
        src_logabsdet: torch.Tensor, # (N,)
        src_field: torch.Tensor,     # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        tspan = self.tspan * (end_time - start_time) + start_time
        tgt_field = self._run_euler(src_coords, src_logabsdet, src_field, tspan)
        return src_coords, src_logabsdet, tgt_field

    def pullback(
        self,
        tgt_coords: torch.Tensor,    # (N, d)
        tgt_logabsdet: torch.Tensor, # (N,)
        tgt_field: torch.Tensor,     # (B, N, C)
        start_time: float,
        end_time: float,
    ):
        tspan = self.tspan.flip(0) * (end_time - start_time) + start_time
        src_field = self._run_euler(tgt_coords, tgt_logabsdet, tgt_field, tspan)
        return tgt_coords, tgt_logabsdet, src_field
