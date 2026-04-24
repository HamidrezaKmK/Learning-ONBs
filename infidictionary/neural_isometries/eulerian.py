from typing import Callable, Dict, Any
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .base import NeuralIsometry
from infidictionary.networks import TimeEvolvingField, SinusoidalTimeEmbedding
from infidictionary.networks.base import _build_mlp
from infidictionary.dictionaries import FourierDictionary


class EulerianIsometry(NeuralIsometry):

    def __init__(
        self,
        coords_dim: int,
        channels_dim: int,
        rank: int,
        base_acceleration: float,
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField],
        gradient_checkpointing: bool = False,
        n_time_freqs: int = 16,
        kr_hidden_dims: tuple = (64, 64),
        atom_shuffling_K: int | None = None,
        atom_selector_hidden_dims: tuple = (64, 64),
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        self.function_field = scalar_field_partial(
            coords_dim=coords_dim,
            output_dim=channels_dim,
            rank=rank,
            time_emb_dim=emb_dim,
        )
        self.coords_dim = coords_dim
        self.channels_dim = channels_dim
        self.rank = rank
        self.gradient_checkpointing = gradient_checkpointing
        self.base_acceleration = base_acceleration
        self._num_steps = 0
        self._model_state_seed: int = 0
        self.register_buffer("tspan", torch.tensor([]), persistent=False)

        # K_R network: time → (R, R)
        self.kr_mlp = _build_mlp(
            emb_dim,
            rank * rank,
            kr_hidden_dims,
            nn.SiLU,
            use_batchnorm=False,
            use_rmsnorm=True,
            bias=True,
        )

        # Optional atom-shuffling: Fourier basis mixing entirely inside EulerianIsometry.
        self.atom_shuffling_K = atom_shuffling_K
        if atom_shuffling_K is not None:
            self.atom_fourier_dict = FourierDictionary(
                domain_dim=coords_dim,
                num_channels=channels_dim,
                p=0.3,
            )
            atom_indices = self.atom_fourier_dict.get_truncated_indices(atom_shuffling_K)
            self.register_buffer("atom_indices", atom_indices)
            n_atoms = len(atom_indices)
            # time → (n_atoms, R, C) selector weights
            self.atom_selector = _build_mlp(
                emb_dim,
                n_atoms * rank * channels_dim,
                atom_selector_hidden_dims,
                nn.SiLU,
                use_batchnorm=False,
                use_rmsnorm=True,
                bias=True,
            )

    def _compute_kr(self, t_emb: torch.Tensor) -> torch.Tensor:
        """(T, emb_dim) → (T, R, R)."""
        return self.kr_mlp(t_emb).view(-1, self.rank, self.rank)

    # ── Cayley step ────────────────────────────────────────────────────────────

    def _cayley_step(
        self,
        t0: float,
        t1: float,
        U: torch.Tensor,           # (N, R, C)
        K_R: torch.Tensor,         # (R, R)
        logabsdet: torch.Tensor,   # (N,)
        values: torch.Tensor,      # (B, N, C)
        acceleration: float,
    ) -> torch.Tensor:             # (B, N, C)
        """Rank-R Cayley isometry on the span of the R network-output functions.

        Skew generator: K = (Δt·a) · U (K_R − K_Rᵀ) U*
        Cayley map: f ↦ (I − K/2)⁻¹(I + K/2) f
        Woodbury reduction: only an R×R system is solved.
        Reversing Δt gives the exact inverse.
        """
        N, R, _ = U.shape
        delta_t = t1 - t0
        exp_lad = logabsdet.exp()  # (N,)

        # B̃ = ½(Δt·a)(K_R − K_Rᵀ)
        B_tilde = 0.5 * (delta_t * acceleration) * (K_R - K_R.transpose(-1, -2))  # (R, R)

        # Gram G[r,s] = ⟨u_r, u_s⟩
        U_w = U * exp_lad[:, None, None]
        G = torch.einsum("nrc,nsc->rs", U_w, U) / N                            # (R, R)

        # c = U* φ
        c = torch.einsum("nrc,bnc,n->br", U, values, exp_lad) / N              # (B, R)

        # y = φ + U (B̃ c)
        Bc = torch.einsum("rs,bs->br", B_tilde, c)                             # (B, R)
        y = values + torch.einsum("nrc,br->bnc", U, Bc)                        # (B, N, C)

        # Solve (I_R − G B̃) z = U* y
        tilde_c = torch.einsum("nrc,bnc,n->br", U, y, exp_lad) / N             # (B, R)
        I_R = torch.eye(R, device=U.device, dtype=U.dtype)
        M = I_R - G @ B_tilde                                                   # (R, R)
        z = torch.linalg.solve(M, tilde_c.transpose(-1, -2)).transpose(-1, -2) # (B, R)

        # φ_new = y + U (B̃ z)
        Bz = torch.einsum("rs,bs->br", B_tilde, z)                             # (B, R)
        return y + torch.einsum("nrc,br->bnc", U, Bz)

    # ── Euler integrator ───────────────────────────────────────────────────────

    def _run_euler(
        self,
        coords: torch.Tensor,    # (N, d)
        logabsdet: torch.Tensor, # (N,)
        f: torch.Tensor,         # (B, N, C)
        tspan: torch.Tensor,     # (T,)
    ) -> torch.Tensor:           # (B, N, C)
        N = f.shape[1]
        use_ckpt = self.gradient_checkpointing and self.training
        t_mid = (tspan[:-1] + tspan[1:]) / 2  # (T-1,)
        t_mid = t_mid.to(f.device)

        # Precompute time embeddings and K_R for all steps (cheap, R×R each).
        all_t_emb = self.time_embedding(t_mid)           # (T-1, emb_dim)
        all_kr = self._compute_kr(all_t_emb)             # (T-1, R, R)

        # Precompute Fourier atom features once (expensive spatial eval, no grad).
        if self.atom_shuffling_K is not None:
            with torch.no_grad():
                all_features = self.atom_fourier_dict.get_atoms(
                    coords.detach(), self.atom_indices
                ).detach().permute(1, 0, 2)              # (N, n_atoms, C)
        else:
            all_features = None

        for i, (t0, t1) in enumerate(zip(tspan[:-1].tolist(), tspan[1:].tolist())):
            K_R = all_kr[i]
            t_emb_step = all_t_emb[i]                   # (emb_dim,)

            def step(v, _t_emb=t_emb_step, _t0=t0, _t1=t1,
                     _K_R=K_R, _feats=all_features):
                t_emb_N = _t_emb[None].expand(N, -1)    # (N, emb_dim)
                U = self.function_field(t_emb_N, coords) # (N, R, C)
                if _feats is not None:
                    n_atoms = _feats.shape[1]
                    w = self.atom_selector(_t_emb[None]).view(
                        n_atoms, self.rank, self.channels_dim
                    )                                    # (n_atoms, R, C)
                    U = U + (w[None] * _feats[:, :, None, :]).sum(1)  # (N, R, C)
                return self._cayley_step(_t0, _t1, U, _K_R, logabsdet, v, self.base_acceleration)

            f = checkpoint(step, f, use_reentrant=False) if use_ckpt else step(f)

        return f

    # ── Public interface ───────────────────────────────────────────────────────

    def shuffle_model_state(self, num_steps: int | None = None):
        if num_steps is not None:
            self._num_steps = num_steps
        if self._num_steps == 0:
            raise ValueError("Call shuffle_model_state with num_steps > 0 first.")
        if self.training:
            t_rand = torch.rand(self._num_steps)
            self.tspan = torch.sort(t_rand)[0]
        else:
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
