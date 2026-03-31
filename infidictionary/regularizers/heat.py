from .base import Regularizer, _eval_pushed_atoms
import torch

from infidictionary.utils import norm2, parallel_inner_product

class DirichletEnergyRegularizer(Regularizer):
    """Regularizer minimising the weighted Dirichlet energy via finite differences.

    For each atom a, estimates:

        E_a = ∫ D(x) ‖∇(Qφ_a)(x)‖² dx

    using stochastic centered finite differences.  For each random unit direction
    δ ~ Uniform(S^{d-1}) and step size h:

        ∇(Qφ_a)(x) · δ  ≈  (Qφ_a(x+hδ) - Qφ_a(x-hδ)) / (2h)

    Since E[(∇f · δ)²] = (1/d)‖∇f‖², the estimator multiplies by d:

        E_a ≈ (d / n_fd_dirs) Σ_k ⟨diff_k, D · diff_k⟩_{L²}

    where diff_k = (Qφ_a(x+hδ_k) - Qφ_a(x-hδ_k)) / (2h).

    This avoids autograd through the pushforward, making it cheaper for large
    numbers of atoms and compatible with non-differentiable pipelines.

    Args:
        domain_sampler:     A ``DomainSampler`` providing quadrature points.
        domain_sample_size: Passed as n_per_dim to domain_sampler.sample.
        material:           A ``Material`` providing D(x) and \rho(x).
        finite_diff_h:      Finite-difference step size h.
        n_fd_dirs:          Number of random directions to average over.
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        material,
        finite_diff_h: float = 1e-3,
        n_fd_dirs: int = 1,
    ):
        super().__init__(domain_sampler, domain_sample_size)
        self.material = material
        self.finite_diff_h = finite_diff_h
        self.n_fd_dirs = n_fd_dirs
        self._diffusivity: torch.Tensor | None = None
        self._mass: torch.Tensor | None = None

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        with torch.no_grad():
            diffusivity, mass = self.material(self._coords)
            self._diffusivity = diffusivity.detach()
            self._mass = mass.detach()

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:

        if self._diffusivity is None:
            raise RuntimeError(
                "DirichletEnergyRegularizer.update_coordinates must be called "
                "before compute_energy."
            )

        tgt_coords = self._coords.to(indices.device)
        N, d = tgt_coords.shape
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        # Integration measure: D(x) \rho(x) dx
        log_D_rho = self._diffusivity.to(device).log() + self._mass.to(device).log()  # (N,)

        de_per_atom = torch.zeros(indices.shape[0], device=device, dtype=dtype)

        for _ in range(self.n_fd_dirs):
            # Random unit direction δ ~ Uniform(S^{d-1})
            delta = torch.randn(N, d, device=device, dtype=dtype)
            delta = delta / delta.norm(dim=-1, keepdim=True)

            coords_plus  = tgt_coords + self.finite_diff_h * delta
            coords_plus  = coords_plus  - torch.floor(coords_plus)   # periodic wrap
            coords_minus = tgt_coords - self.finite_diff_h * delta
            coords_minus = coords_minus - torch.floor(coords_minus)

            _, qphi_plus,  _ = _eval_pushed_atoms(
                neural_isometry, initial_dictionary, coords_plus,  indices, pushforward_kwargs
            )
            _, qphi_minus, _ = _eval_pushed_atoms(
                neural_isometry, initial_dictionary, coords_minus, indices, pushforward_kwargs
            )

            # Centered finite-difference directional derivative: (A, N, C)
            diff = (qphi_plus - qphi_minus) / (2.0 * self.finite_diff_h)

            de_per_atom = de_per_atom + parallel_inner_product(diff, diff, logabsdet=log_D_rho)

        # E[(∇f · δ)²] = (1/d)‖∇f‖²  →  multiply by d to recover full energy
        return de_per_atom * (d / self.n_fd_dirs)


class HeatQuadraticFormRegularizer(Regularizer):
    """Regularizer returning the *negative* heat-kernel quadratic form, with
    an optional masking penalty for atoms outside the material support.

    The heat-kernel quadratic form

        ⟨Qφ_a, P̃_t Qφ_a⟩_\rho

    is positive and bounded in (0, 1].  This class returns its *negative* so
    that minimising the regularizer energy corresponds to maximising the
    quadratic form (i.e. finding the vibrational modes of the material).

    P̃_t is the semigroup of \rho⁻¹L approximated by a single Euler–Maruyama
    step of the SDE ``dX ≈ √(2 D(x)/\rho(x)) dW`` with periodic wrapping:

        P̃_t Qφ(x) ≈ E_Z[ Qφ(x + √(2D(x)t/\rho(x)) Z) ],   Z ~ N(0, I)

    When ``masking_weight > 0``, an additional masking term is added to the
    energy: ``masking_weight * ‖Qφ_a · 𝟙_outside‖²``, penalising atom values
    in regions with negligible mass (``mass < mass_threshold``).

    Args:
        domain_sampler:      A ``DomainSampler`` providing quadrature points.
        domain_sample_size:  Passed as n_per_dim to domain_sampler.sample.
        material:            A ``Material`` providing D(x) and \rho(x).
        smoothing_t:         Diffusion time t in P̃_t = e^{-t \rho⁻¹ L}.
        n_diffusion_samples: Number of MC draws for the diffusion estimate.
        masking_weight:      Weight for the outside-mask penalty (0 disables it).
        mass_threshold:      Points with ``mass < mass_threshold`` are outside.
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        material,
        smoothing_t: float = 1.0,
        n_diffusion_samples: int = 4,
        masking_weight: float = 0.0,
        mass_threshold: float = 0.5,
    ):
        super().__init__(domain_sampler, domain_sample_size)
        self.material = material
        self.smoothing_t = smoothing_t
        self.n_diffusion_samples = n_diffusion_samples
        self.masking_weight = masking_weight
        self.mass_threshold = mass_threshold
        self._diffusivity: torch.Tensor | None = None
        self._mass: torch.Tensor | None = None
        self._outside: torch.Tensor | None = None

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        with torch.no_grad():
            diffusivity, mass = self.material(self._coords)
            self._diffusivity = diffusivity.detach()
            self._mass = mass.detach()
            if self.masking_weight > 0.0:
                self._outside = (mass < self.mass_threshold).float().detach()

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:

        if self._diffusivity is None:
            raise RuntimeError(
                "HeatQuadraticFormRegularizer.update_coordinates must be called "
                "before compute_energy."
            )

        device = indices.device
        tgt_coords = self._coords.to(device)
        N, d = tgt_coords.shape
        dtype = tgt_coords.dtype

        diffusivity = self._diffusivity.to(device)
        mass = self._mass.to(device)

        src_coords = self._pulled_back_coords.to(device)
        src_logabsdet = self._pulled_back_logabsdet.to(device)

        # Base: Qφ_a(x)
        _, qphi_base, _ = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs,
            src_coords=src_coords, src_logabsdet=src_logabsdet,
        )

        # MC estimate of P̃_t Qφ(x) = E_Z[ Qφ(x + √(2D(x)t/\rho(x)) Z) ]
        diffusion_std = torch.sqrt(
            2.0 * diffusivity * self.smoothing_t / mass
        ).unsqueeze(-1)  # (N, 1)

        qphi_diffused = torch.zeros_like(qphi_base)
        for _ in range(self.n_diffusion_samples):
            Z = torch.randn(N, d, device=device, dtype=dtype)
            coords_shifted = tgt_coords + diffusion_std * Z
            coords_shifted = coords_shifted - torch.floor(coords_shifted)  # periodic wrap
            _, qphi_shifted, _ = _eval_pushed_atoms(
                neural_isometry, initial_dictionary, coords_shifted, indices, pushforward_kwargs
            )
            qphi_diffused = qphi_diffused + qphi_shifted
        qphi_diffused = qphi_diffused / self.n_diffusion_samples

        log_mass = torch.log(mass)
        qform = parallel_inner_product(qphi_base, qphi_diffused, logabsdet=log_mass)

        # Negative so that minimising energy = maximising the quadratic form.
        energy = -qform

        if self.masking_weight > 0.0:
            outside = self._outside.to(device)[None, :, None]  # (1, N, 1)
            energy = energy + self.masking_weight * norm2(qphi_base * outside)

        return energy
