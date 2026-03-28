from .base import PushforwardRegularizer, Regularizer, _base_push_cache
import torch

class MaskingRegularizer(PushforwardRegularizer):
    """Penalises atom values outside the material mask: E_a = ‖Qφ_a · 𝟙_outside‖².

    Applied uniformly across all channels.  Useful for suppressing pushed-forward
    atoms in regions with negligible mass (e.g. outside a portrait or shape).

    Args:
        material:        A ``Material`` used to derive the mask via its mass map.
        mass_threshold:  Points with ``mass < mass_threshold`` are *outside*.
    """

    def __init__(self, material, mass_threshold: float = 0.5):
        self.material = material
        self.mass_threshold = mass_threshold
        self._outside: torch.Tensor | None = None

    def update_coordinates(self, coords: torch.Tensor) -> None:
        with torch.no_grad():
            _, mass = self.material(coords)
            self._outside = (mass < self.mass_threshold).float().detach()

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        from infidictionary.utils import norm2

        if self._outside is None:
            raise RuntimeError(
                "MaskingRegularizer.update_coordinates must be called before compute_energy."
            )
        outside = self._outside[None, :, None]  # (1, N, 1) — broadcast over atoms and channels
        return norm2(pushed * outside)           # (A,)


class HeatQuadraticFormRegularizer(Regularizer):
    """Regularizer returning the *negative* heat-kernel quadratic form.

    The heat-kernel quadratic form

        ⟨Qφ_a, P̃_t Qφ_a⟩_ρ

    is positive and bounded in (0, 1].  This class returns its *negative* so
    that minimising the regularizer energy corresponds to maximising the
    quadratic form (i.e. finding the vibrational modes of the material).

    P̃_t is the semigroup of ρ⁻¹L approximated by a single Euler–Maruyama
    step of the SDE ``dX ≈ √(2 D(x)/ρ(x)) dW`` with periodic wrapping:

        P̃_t Qφ(x) ≈ E_Z[ Qφ(x + √(2D(x)t/ρ(x)) Z) ],   Z ~ N(0, I)

    Args:
        material:            A ``Material`` providing D(x) and ρ(x).
        smoothing_t:         Diffusion time t in P̃_t = e^{-t ρ⁻¹ L}.
        n_diffusion_samples: Number of MC draws for the diffusion estimate.
    """

    def __init__(self, material, smoothing_t: float = 1.0, n_diffusion_samples: int = 4):
        self.material = material
        self.smoothing_t = smoothing_t
        self.n_diffusion_samples = n_diffusion_samples
        self._diffusivity: torch.Tensor | None = None
        self._mass:        torch.Tensor | None = None

    def update_coordinates(self, coords: torch.Tensor) -> None:
        with torch.no_grad():
            diffusivity, mass = self.material(coords)
            self._diffusivity = diffusivity.detach()
            self._mass = mass.detach()

    def compute_energy(
        self, neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
    ) -> torch.Tensor:
        from infidictionary.utils import parallel_inner_product

        if self._diffusivity is None:
            raise RuntimeError(
                "HeatQuadraticFormRegularizer.update_coordinates must be called "
                "before compute_energy."
            )

        N, d = tgt_coords.shape
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        # Base: Qφ_a(x)
        _, qphi_base, _ = _base_push_cache.get_or_compute(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
        )

        # MC estimate of P̃_t Qφ(x) = E_Z[ Qφ(x + √(2D(x)t/ρ(x)) Z) ]
        diffusion_std = torch.sqrt(
            2.0 * self._diffusivity * self.smoothing_t / self._mass
        ).unsqueeze(-1)  # (N, 1)

        qphi_diffused = torch.zeros_like(qphi_base)
        for _ in range(self.n_diffusion_samples):
            Z = torch.randn(N, d, device=device, dtype=dtype)
            coords_shifted = tgt_coords + diffusion_std * Z
            coords_shifted = coords_shifted - torch.floor(coords_shifted)  # periodic wrap
            _, qphi_shifted, _ = _base_push_cache.get_or_compute(
                neural_isometry, initial_dictionary, coords_shifted, indices, pushforward_kwargs
            )
            qphi_diffused = qphi_diffused + qphi_shifted
        qphi_diffused = qphi_diffused / self.n_diffusion_samples

        log_mass = torch.log(self._mass)
        qform = parallel_inner_product(qphi_base, qphi_diffused, logabsdet=log_mass)

        # Return the *negative* so that minimising energy = maximising the quadratic form.
        return -qform
