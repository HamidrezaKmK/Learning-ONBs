from .base import Regularizer
import torch

class NTKRegularizer(Regularizer):
    """NTK quadratic-form regularizer.

    Maximises  Σ_a ⟨Qφ_a, K Qφ_a⟩ where K is the empirical Neural Tangent
    Kernel of a fixed ``ntk_model``,

        K(x, y) = ⟨∇_θ f(x; θ), ∇_θ f(y; θ)⟩  =  J(θ)·J(θ)^T

    Can compute via two equivalent methods:

    1. **pullback-then-compute** (default):
       Uses the identity ⟨Qφ_a, K Qφ_a⟩ = ‖∇_θ ⟨Q* f_θ, φ_a⟩‖²
       Computes via Jacobian of inner products; K never explicitly formed.
       Gradients flow through via ``create_graph=True``.

    2. **pushforward-then-compute**:
       Directly computes ‖E_x[(Qφ_a)(x) ∇_θ f_θ(x)]‖² by evaluating
       gradients at points and weighting by pushed-forward atoms.
       More direct but requires per-atom gradient computation.

    Args:
        domain_sampler:         A ``DomainSampler`` providing quadrature points.
        domain_sample_size:     Passed as n_per_dim to domain_sampler.sample.
        ntk_model:              A ``NeuralField`` whose parameters define the NTK.
                                The model is kept in eval mode and its weights are
                                not updated.
        ntk_model_weights_path: Optional path to a checkpoint to load into ntk_model.
                                The checkpoint must contain ``model_state_dict``.
        method:                 "pullback-then-compute" or "pushforward-then-compute".
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        ntk_model,
        ntk_model_weights_path: str | None = None,
        method: str = "pullback-then-compute",
    ) -> None:
        super().__init__(domain_sampler, domain_sample_size)
        self.ntk_model = ntk_model
        if ntk_model_weights_path is not None:
            ckpt = torch.load(ntk_model_weights_path, weights_only=False, map_location="cpu")
            self.ntk_model.load_state_dict(ckpt["model_state_dict"])
        self.ntk_model.eval()
        if method not in ("pullback-then-compute", "pushforward-then-compute"):
            raise ValueError(f"method must be 'pullback-then-compute' or 'pushforward-then-compute', got {method}")
        self.method = method

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        if self.method == "pullback-then-compute":
            return self._compute_energy_pullback_then(
                neural_isometry, initial_dictionary, indices, pushforward_kwargs
            )
        else:
            return self._compute_energy_pushforward_then(
                neural_isometry, initial_dictionary, indices, pushforward_kwargs
            )

    def _compute_energy_pullback_then(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        """Compute via Jacobian: ⟨Qφ, KQφ⟩ = ‖∇_θ ⟨Q* f_θ, φ_a⟩‖²"""
        from torch.func import functional_call
        from infidictionary.utils import pairwise_inner_product

        if self._coords is None:
            raise RuntimeError(
                "NTKRegularizer.update_coordinates must be called before compute_energy."
            )

        device = indices.device
        tgt_coords = self._coords.to(device)
        N = tgt_coords.shape[0]
        dtype = tgt_coords.dtype

        self.ntk_model.to(device)

        src_coords = self._pulled_back_coords.to(device)
        src_logabsdet = self._pulled_back_logabsdet.to(device)
        tgt_logabsdet = torch.zeros(N, device=device, dtype=dtype)

        A = indices.shape[0]
        phi_src = initial_dictionary.get_atoms(src_coords, indices)  # (A, N, C)

        ntk_param_names = [name for name, _ in self.ntk_model.named_parameters()]

        def c_from_ntk_params(*params):
            """Inner product ⟨Q* f_θ, φ_a⟩ as a function of θ — shape (A,)."""
            ntk_dict = dict(zip(ntk_param_names, params))
            f_vals = functional_call(self.ntk_model, ntk_dict, (tgt_coords,))
            if f_vals.dim() == 1:
                f_vals = f_vals.unsqueeze(-1)  # (N,) → (N, 1)
            _, _, qsf = neural_isometry.pullback(
                tgt_coords=tgt_coords,
                tgt_logabsdet=tgt_logabsdet,
                tgt_field=f_vals.unsqueeze(0),  # (1, N, C)
                **pushforward_kwargs,
            )
            return pairwise_inner_product(qsf, phi_src, src_logabsdet).squeeze(0)

        J_tuple = torch.autograd.functional.jacobian(
            c_from_ntk_params,
            tuple(self.ntk_model.parameters()),
            create_graph=True,
            vectorize=True,
        )
        qf = sum(j.reshape(A, -1).pow(2).sum(-1) for j in J_tuple)

        return -qf

    def _compute_energy_pushforward_then(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        """Compute via direct gradients: ‖∇_θ ⟨f_θ, Qφ_a⟩‖² for each atom.

        Pushes φ_a forward through Q to get Qφ_a at target coords, then for
        each atom computes ⟨f_θ, Qφ_a⟩ and squares the norm of its gradient
        w.r.t. NTK params. By the duality ⟨f_θ, Qφ_a⟩ = ⟨Q*f_θ, φ_a⟩ for
        unitary Q, this is mathematically equivalent to the pullback variant.
        """
        from infidictionary.utils import pairwise_inner_product

        if self._coords is None:
            raise RuntimeError(
                "NTKRegularizer.update_coordinates must be called before compute_energy."
            )

        device = indices.device
        tgt_coords = self._coords.to(device)
        dtype = tgt_coords.dtype

        self.ntk_model.to(device)

        src_coords = self._pulled_back_coords.to(device)
        src_logabsdet = self._pulled_back_logabsdet.to(device)

        A = indices.shape[0]
        phi_src = initial_dictionary.get_atoms(src_coords, indices)  # (A, N, C)

        tgt_pf, tgt_logabsdet, pushed = neural_isometry.pushforward(
            src_coords=src_coords,
            src_logabsdet=src_logabsdet,
            src_field=phi_src,
            **pushforward_kwargs,
        )  # pushed: (A, N, C) — Qφ at tgt_pf

        f_vals = self.ntk_model(tgt_pf)  # (N,) or (N, C)
        if f_vals.dim() == 1:
            f_vals = f_vals.unsqueeze(-1)
        f_vals_b = f_vals.unsqueeze(0)  # (1, N, C)

        qf = torch.zeros(A, device=device, dtype=dtype)

        for a in range(A):
            inner = pairwise_inner_product(
                f_vals_b, pushed[a:a + 1], tgt_logabsdet
            ).squeeze()

            grads = torch.autograd.grad(
                inner,
                self.ntk_model.parameters(),
                create_graph=True,
                allow_unused=True,
            )

            grad_norm_sq = sum(
                (g.reshape(-1) ** 2).sum()
                for g in grads
                if g is not None
            )
            qf[a] = grad_norm_sq

        return -qf
