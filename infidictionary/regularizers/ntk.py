from .base import Regularizer
import torch

class NTKRegularizer(Regularizer):
    """NTK quadratic-form regularizer.

    Maximises  Σ_a ⟨Qφ_a, K Qφ_a⟩  where K is the Neural Tangent Kernel of a
    fixed ``ntk_model``, using the identity

        ⟨Qφ_a, K Qφ_a⟩  =  ‖∇_θ ⟨Q* f_θ, φ_a⟩‖²

    so K is never explicitly formed.  Gradients flow through the Jacobian to
    the isometry parameters via ``create_graph=True``.

    Args:
        domain_sampler:         A ``DomainSampler`` providing quadrature points.
        domain_sample_size:     Passed as n_per_dim to domain_sampler.sample.
        ntk_model:              A ``NeuralField`` whose parameters define the NTK.
                                The model is kept in eval mode and its weights are
                                not updated.
        ntk_model_weights_path: Optional path to a checkpoint to load into ntk_model.
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        ntk_model,
        ntk_model_weights_path: str | None = None,
    ) -> None:
        super().__init__(domain_sampler, domain_sample_size)
        self.ntk_model = ntk_model
        if ntk_model_weights_path is not None:
            ckpt = torch.load(ntk_model_weights_path, weights_only=False, map_location="cpu")
            self.ntk_model.load_state_dict(ckpt["model_state_dict"])
        self.ntk_model.eval()

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        from torch.func import functional_call
        from infidictionary.neural_isometries import EulerianIsometry
        from infidictionary.utils import pairwise_inner_product

        tgt_coords = self._coords.to(indices.device)
        N = tgt_coords.shape[0]
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        self.ntk_model.to(device)

        # Pull back tgt_coords → src_coords so we can evaluate source atoms.
        # Gradients are not needed through this step.
        if isinstance(neural_isometry, EulerianIsometry):
            src_coords = tgt_coords
            src_logabsdet = torch.zeros(N, device=device, dtype=dtype)
            tgt_logabsdet = torch.zeros(N, device=device, dtype=dtype)
        else:
            with torch.no_grad():
                src_coords, src_logabsdet, _ = neural_isometry.pullback(
                    tgt_coords=tgt_coords,
                    tgt_logabsdet=torch.zeros(N, device=device, dtype=dtype),
                    tgt_field=torch.zeros(1, N, 1, device=device, dtype=dtype),
                    **pushforward_kwargs,
                )
            src_coords = src_coords.detach()
            src_logabsdet = src_logabsdet.detach()
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
            # qsf: (1, N, C); pairwise with (A, N, C) → (1, A) → (A,)
            return pairwise_inner_product(qsf, phi_src, src_logabsdet).squeeze(0)

        # Jacobian of shape (A, *param_shape) per parameter tensor.
        # create_graph=True lets gradients flow through to the isometry.
        J_tuple = torch.autograd.functional.jacobian(
            c_from_ntk_params,
            tuple(self.ntk_model.parameters()),
            create_graph=True,
            vectorize=True,
        )
        qf = sum(j.reshape(A, -1).pow(2).sum(-1) for j in J_tuple)  # (A,)

        # Return the *negative* so that minimising energy = maximising the NTK QF.
        return -qf
