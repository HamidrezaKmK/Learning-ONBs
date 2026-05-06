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
        scalar_field_partial: Callable[[Dict[str, Any]], TimeEvolvingField] | None = None,
        gradient_checkpointing: bool = False,
        n_time_freqs: int = 16,
        kr_hidden_dims: tuple = (64, 64),
        atom_shuffling_K: int | None = None,
        atom_selector_hidden_dims: tuple = (64, 64),
        use_power_law: bool = False,
    ):
        super().__init__()

        # At least one source of spatial functions must be configured: either the
        # MLP (`scalar_field_partial`) or the Fourier-feature atom selector
        # (`atom_shuffling_K`). With both unset the rank-R generator collapses
        # to zero and the isometry would just be the identity.
        if scalar_field_partial is None and atom_shuffling_K is None:
            raise ValueError(
                "EulerianIsometry needs at least one source of spatial functions: "
                "set `scalar_field_partial` (MLP) and/or `atom_shuffling_K` "
                "(Fourier-feature atom selector). Both were None."
            )

        self.time_embedding = SinusoidalTimeEmbedding(n_time_freqs)
        emb_dim = self.time_embedding.out_dim

        # Spatial generator U(t, x): (N, emb_dim) x (N, d) → (N, R, C).
        # Optional MLP path; when None, U comes purely from the atom-selector
        # residual (a time-varying linear combination of Fourier atoms).
        self._use_function_field = scalar_field_partial is not None
        if self._use_function_field:
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
        self.use_power_law = use_power_law
        if atom_shuffling_K is not None:
            self.atom_fourier_dict = FourierDictionary(
                domain_dim=coords_dim,
                num_channels=1,
                steepness=2.0,
                tiebreak_eps=0.0,   # this dict is only used for atom features, no PMF needed
            )
            atom_indices = self.atom_fourier_dict.get_truncated_indices(atom_shuffling_K)
            self.register_buffer("atom_indices", atom_indices)

            if self.use_power_law:
                # Power-law (1 / ||k||²) prior on the atom-selector output along the
                # n_atoms axis. Natural images have a 1/k² power spectrum, so this
                # is an inductive bias that gives low-frequency atoms more amplitude
                # by default and lets the MLP only need to "deviate from prior".
                # The DC mode (||k||=0) and shell-1 atoms (||k||=1) get weight 1; a
                # ||k||=10 corner atom gets weight 1/100.
                spatial_k = atom_indices[:, :coords_dim].to(torch.get_default_dtype())  # (n_atoms, d)
                k_norm_sq = spatial_k.pow(2).sum(dim=-1)                                  # (n_atoms,)
                atom_weights = 1.0 / k_norm_sq.clamp(min=1.0)                             # (n_atoms,)
                self.register_buffer("atom_weights", atom_weights)

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

    # ── Assemble the full generator field U(t, x) ──────────────────────────────

    def _assemble_U(
        self,
        t_emb_N: torch.Tensor,                  # (N, emb_dim) — same time on every row
        coords: torch.Tensor,                   # (N, d)
        precomputed_features: torch.Tensor | None = None,  # (N, n_atoms, C)
    ) -> torch.Tensor:                          # (N, R, C)
        """Spatial generator field that drives the rank-R Cayley rotation.

        ``U(t, x) = function_field(t, x)  +  Σ_a w_a(t) · φ_a(x)``

        — the MLP path (``function_field``) and the atom-selector residual
        (a time-varying linear combination of Fourier atoms ``φ_a``). Either
        path may be disabled at construction time, but at least one is always
        present (enforced in ``__init__``). All callers — the integrator and
        any visualisation — must use this assembled field, since it's the
        actual quantity that defines the rotation.

        ``precomputed_features`` lets ``_run_euler`` avoid redundant Fourier
        evaluations across the integration timesteps.
        """
        N = coords.shape[0]
        if self._use_function_field:
            U = self.function_field(t_emb_N, coords)                            # (N, R, C)
        else:
            U = torch.zeros(
                N, self.rank, self.channels_dim,
                device=coords.device, dtype=coords.dtype,
            )

        if self.atom_shuffling_K is not None:
            if precomputed_features is None:
                with torch.no_grad():
                    feats = self.atom_fourier_dict.get_atoms(
                        coords.detach(), self.atom_indices
                    ).detach().permute(1, 0, 2).repeat(1, 1, self.channels_dim) # (N, n_atoms, C)
            else:
                feats = precomputed_features
            n_atoms = feats.shape[1]
            # atom_selector depends only on time; take the first row of t_emb_N.
            w = self.atom_selector(t_emb_N[:1]).view(
                n_atoms, self.rank, self.channels_dim,
            )                                                                    # (n_atoms, R, C)
            if self.use_power_law:
                # 1 / ||k||² inductive bias along the n_atoms axis (natural-image
                # power-spectrum prior — low-freq atoms dominate by default).
                w = w * self.atom_weights[:, None, None]                             # (n_atoms, R, C)
            U = U + (w[None] * feats[:, :, None, :]).sum(1)                      # (N, R, C)

        return U

    # ── Cayley step (cross-channel rank-R) ─────────────────────────────────────

    def _joint_cayley_step(
        self,
        t0: float,
        t1: float,
        U: torch.Tensor,                   # (N, R, C)
        K_R: torch.Tensor,                 # (R, R)
        logabsdet: torch.Tensor,           # (N,)
        values: torch.Tensor,              # (B, N, C)
        acceleration: float,
    ) -> torch.Tensor:                     # (B, N, C)
        """Cross-channel rank-R Cayley step.

        Generator:

            ⟨U, f⟩_r     = (1/N) Σ_{n,c} U[n,r,c] · f[n,c] · e^{l_n}     ← sums over BOTH n and c
            (K f)(x)_c   = Σ_r U(x)_{r,c} · (A_R · ⟨U, f⟩)_r              ← rank-R, multi-channel

        Cayley map  f ↦ (I − βK)⁻¹(I + βK) f,  β = ½ Δt · a.

        Because U* uses the full L² inner product (summed over channels),
        K is a single rank-R skew operator on the multi-channel function
        space — channels are coupled directly through U(x), so the Cayley flow
        can rotate energy between channels via U alone.

        Single Woodbury R × R solve:

            B_op  = β(K_R − K_Rᵀ)                            (R, R)  skew
            G     = (1/N) Σ_n U[n,r,c]·U[n,s,c]·e^{l_n}      (R, R)

            y_pre = f + U · B_op · ⟨U, f⟩
            z'    = (I_R − G B_op)⁻¹ ⟨U, y_pre⟩              (R × R solve)
            f_new = y_pre + U · B_op · z'

        Replacing Δt by −Δt gives the exact inverse — pullback inverts
        pushforward up to linear-solve precision.
        """
        N, R, C = U.shape
        beta    = 0.5 * (t1 - t0) * acceleration
        exp_lad = logabsdet.exp()                                                  # (N,)
        B_op    = beta * (K_R - K_R.transpose(-1, -2))                             # (R, R) scaled skew

        # ── Step 1: y_pre = f + U · B_op · ⟨U, f⟩  (cross-channel) ───────────
        # ⟨U, f⟩_r = (1/N) Σ_{n,c} U[n,r,c] · f[n,c] · e^{l_n}
        c_proj = torch.einsum("nrc,bnc,n->br", U, values, exp_lad) / N             # (B, R)
        Bc     = torch.einsum("rs,bs->br", B_op, c_proj)                           # (B, R)
        y_pre  = values + torch.einsum("nrc,br->bnc", U, Bc)                       # (B, N, C)

        # ── Step 2: ⟨U, y_pre⟩  (cross-channel) ──────────────────────────────
        tilde_c = torch.einsum("nrc,bnc,n->br", U, y_pre, exp_lad) / N             # (B, R)

        # ── Step 3: Gram  G = U* U  (R × R, cross-channel) ───────────────────
        G = torch.einsum("nrc,nsc,n->rs", U, U, exp_lad) / N                       # (R, R)

        # ── Step 4: solve  (I_R − G · B_op) z' = ⟨U, y_pre⟩  ─────────────────
        I_R  = torch.eye(R, device=U.device, dtype=U.dtype)
        Msys = I_R - G @ B_op                                                      # (R, R)
        z_p  = torch.linalg.solve(
            Msys, tilde_c.transpose(-1, -2),
        ).transpose(-1, -2)                                                        # (B, R)
        z    = torch.einsum("rs,bs->br", B_op, z_p)                                # (B, R) = B_op z'

        # ── Step 5: f_new = y_pre + U · z ────────────────────────────────────
        return y_pre + torch.einsum("nrc,br->bnc", U, z)                           # (B, N, C)

    # ── Implicit backward-Euler step (stable but dissipative) ─────────────────

    def _backward_euler_step(
        self,
        t0: float,
        t1: float,
        U: torch.Tensor,                   # (N, R, C)
        K_R: torch.Tensor,                 # (R, R)
        logabsdet: torch.Tensor,           # (N,)
        values: torch.Tensor,              # (B, N, C)
        acceleration: float,
    ) -> torch.Tensor:                     # (B, N, C)
        """Implicit backward-Euler step  (I − Δt · a · K) f_new = f.

        Same generator K as `_joint_cayley_step` (rank-R cross-channel U);
        same Woodbury R × R structure. Differences from Cayley:

          * No (I + βK) factor multiplying f on the right; the implicit
            inverse is applied to f directly.
          * Energy is monotonically dissipated by a factor
            1/√(1 + (Δt·a·σ_K)²) per step (singular value σ_K of K), so
            backward-Euler is unconditionally stable for skew K — it will
            never blow up the way forward-Euler does — but ‖f‖ shrinks.
        """
        N, R, C = U.shape
        dt_a    = (t1 - t0) * acceleration
        exp_lad = logabsdet.exp()                                                  # (N,)
        J       = K_R - K_R.transpose(-1, -2)                                      # (R, R) skew

        # ⟨U, f⟩  (cross-channel)
        Uf = torch.einsum("nrc,bnc,n->br", U, values, exp_lad) / N                 # (B, R)

        # Gram  G = U* U  (R × R)
        G = torch.einsum("nrc,nsc,n->rs", U, U, exp_lad) / N                       # (R, R)

        # Solve  (I_R − Δt·a · G · J) g = ⟨U, f⟩
        I_R  = torch.eye(R, device=U.device, dtype=U.dtype)
        Msys = I_R - dt_a * (G @ J)                                                # (R, R)
        g    = torch.linalg.solve(
            Msys, Uf.transpose(-1, -2),
        ).transpose(-1, -2)                                                        # (B, R)
        Jg   = torch.einsum("rs,bs->br", J, g)                                     # (B, R)

        # f_new = f + Δt·a · U · J · g
        return values + dt_a * torch.einsum("nrc,br->bnc", U, Jg)                  # (B, N, C)

    # ── Naive forward-Euler step (NOT structure-preserving) ────────────────────

    def _naive_euler_step(
        self,
        t0: float,
        t1: float,
        U: torch.Tensor,                   # (N, R, C)
        K_R: torch.Tensor,                 # (R, R)
        logabsdet: torch.Tensor,           # (N,)
        values: torch.Tensor,              # (B, N, C)
        acceleration: float,
    ) -> torch.Tensor:                     # (B, N, C)
        """Plain explicit-Euler step  f ↦ f + Δt · a · K f.

        Same generator K as `_joint_cayley_step` (rank-R cross-channel U),
        but applied directly without the Cayley factorisation. The step is
        **not** norm-preserving — included as a baseline to demonstrate
        that the Cayley parametrisation is what keeps the flow on the
        isometry manifold.
        """
        N = U.shape[0]
        dt_a    = (t1 - t0) * acceleration
        exp_lad = logabsdet.exp()                                                  # (N,)
        J       = K_R - K_R.transpose(-1, -2)                                      # (R, R) skew

        # Cross-channel rank-R part:  K f = U · J · ⟨U, f⟩
        c_proj = torch.einsum("nrc,bnc,n->br", U, values, exp_lad) / N             # (B, R)
        Jc     = torch.einsum("rs,bs->br", J, c_proj)                              # (B, R)
        K_f    = torch.einsum("nrc,br->bnc", U, Jc)                                # (B, N, C)

        return values + dt_a * K_f                                                 # (B, N, C)

    # ── Euler integrator ───────────────────────────────────────────────────────

    def _run_euler(
        self,
        coords: torch.Tensor,    # (N, d)
        logabsdet: torch.Tensor, # (N,)
        f: torch.Tensor,         # (B, N, C)
        tspan: torch.Tensor,     # (T,)
        method: str = "cayley",  # 'cayley' | 'backward-euler' | 'forward-euler'
    ) -> torch.Tensor:           # (B, N, C)
        N = f.shape[1]
        use_ckpt = self.gradient_checkpointing and self.training
        t_mid = (tspan[:-1] + tspan[1:]) / 2  # (T-1,)
        t_mid = t_mid.to(f.device)

        # Precompute time embeddings and K_R for all steps (cheap, RxR each).
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

                # Combined spatial generator U(t, x) = MLP path + atom-selector residual.
                U = self._assemble_U(t_emb_N, coords, precomputed_features=_feats)

                # Dispatch: structure-preserving Cayley | implicit backward-Euler
                # | naive forward-Euler.
                if method == "cayley":
                    return self._joint_cayley_step(
                        _t0, _t1, U, _K_R, logabsdet, v, self.base_acceleration
                    )
                elif method == "backward-euler":
                    return self._backward_euler_step(
                        _t0, _t1, U, _K_R, logabsdet, v, self.base_acceleration
                    )
                elif method == "forward-euler":
                    return self._naive_euler_step(
                        _t0, _t1, U, _K_R, logabsdet, v, self.base_acceleration
                    )
                else:
                    raise ValueError(f"unknown method: {method!r}")

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
        method: str = "cayley",      # 'cayley' | 'backward-euler' | 'forward-euler'
    ):
        tspan = self.tspan * (end_time - start_time) + start_time
        tgt_field = self._run_euler(src_coords, src_logabsdet, src_field, tspan, method=method)
        return src_coords, src_logabsdet, tgt_field

    def pullback(
        self,
        tgt_coords: torch.Tensor,    # (N, d)
        tgt_logabsdet: torch.Tensor, # (N,)
        tgt_field: torch.Tensor,     # (B, N, C)
        start_time: float,
        end_time: float,
        method: str = "cayley",      # 'cayley' | 'backward-euler' | 'forward-euler'
    ):
        tspan = self.tspan.flip(0) * (end_time - start_time) + start_time
        src_field = self._run_euler(tgt_coords, tgt_logabsdet, tgt_field, tspan, method=method)
        return tgt_coords, tgt_logabsdet, src_field
