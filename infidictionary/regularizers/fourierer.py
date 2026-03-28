
from .base import PushforwardRegularizer, Regularizer, _base_push_cache
import torch

class FouriererMasking(PushforwardRegularizer):
    """Region-masking regularizer for two-channel Fourier dictionaries.

    For atoms with multiindex [..., 0]: penalises both channels outside the
    portrait mass (inside-domain atoms should vanish outside).
    For atoms with multiindex [..., 1]: penalises both channels inside the
    portrait mass (outside-domain atoms should vanish inside).

    Args:
        material:        A ``Material`` used to derive the mask via its mass.
        mass_threshold:  Points with ``mass >= mass_threshold`` are *inside*.
    """

    def __init__(self, material, mass_threshold: float = 0.5):
        self.material = material
        self.mass_threshold = mass_threshold
        self._inside:  torch.Tensor | None = None
        self._outside: torch.Tensor | None = None

    def update_coordinates(self, coords: torch.Tensor) -> None:
        with torch.no_grad():
            _, mass = self.material(coords)
            self._inside  = (mass >= self.mass_threshold).float().detach()
            self._outside = 1.0 - self._inside

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices):
        from infidictionary.utils import norm2

        if self._inside is None:
            raise RuntimeError(
                "FouriererMasking.update_coordinates must be called before compute_energy."
            )

        inside  = self._inside[None, :, None]   # (1, N, 1) — broadcast over A, C
        outside = self._outside[None, :, None]

        ch0 = (indices[:, -1] == 0)  # (A,)
        e_ch0 = norm2(pushed * outside)  # both channels zero outside for c=0
        e_ch1 = norm2(pushed * inside)   # both channels zero inside  for c=1
        
        diff_ch = pushed[:, :, 0:1] - pushed[:, :, 1:2]  # (A, N, 1)

        return torch.where(ch0, e_ch0, e_ch1) + norm2(diff_ch)


class FouriererVibration(Regularizer):
    """Heat-kernel vibration regularizer for two-channel Fourier dictionaries.

    For atoms with multiindex [..., 0]: maximises the heat-kernel quadratic form
    on the *inside* domain (ρ_in from the material).
    For atoms with multiindex [..., 1]: maximises the heat-kernel quadratic form
    on the *outside* domain, using the inverted mass ρ_out = (1 − ρ + ε).clamp(ε).

    Use together with ``FouriererMasking`` and ``FouriererChannelEquality`` for
    the full Fourierer vibration regularization scheme.

    Args:
        material:            A ``Material`` providing D(x) and ρ(x).
        smoothing_t:         Diffusion time t in P̃_t.
        n_diffusion_samples: Number of MC draws for the diffusion estimate.
        mass_threshold:      Points with mass >= mass_threshold are *inside*.
        exterior_mass_eps:   Floor added to (1 − mass) for the outside mass.
    """

    def __init__(
        self,
        material,
        smoothing_t: float = 1.0,
        n_diffusion_samples: int = 4,
        mass_threshold: float = 0.5,
        exterior_mass_eps: float = 1e-4,
    ):
        self.material = material
        self.smoothing_t = smoothing_t
        self.n_diffusion_samples = n_diffusion_samples
        self.mass_threshold = mass_threshold
        self.exterior_mass_eps = exterior_mass_eps
        self._diffusivity: torch.Tensor | None = None
        self._mass_in:     torch.Tensor | None = None
        self._mass_out:    torch.Tensor | None = None

    def update_coordinates(self, coords: torch.Tensor) -> None:
        with torch.no_grad():
            diffusivity, mass = self.material(coords)
            self._diffusivity = diffusivity.detach()
            self._mass_in     = mass.detach()
            self._mass_out    = (1.0 - mass + self.exterior_mass_eps).clamp(
                min=self.exterior_mass_eps
            ).detach()

    def compute_energy(
        self, neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
    ) -> torch.Tensor:
        from infidictionary.utils import parallel_inner_product

        if self._diffusivity is None:
            raise RuntimeError(
                "FouriererVibration.update_coordinates must be called before compute_energy."
            )

        N, d = tgt_coords.shape
        device = tgt_coords.device
        dtype  = tgt_coords.dtype

        std_in  = torch.sqrt(
            2.0 * self._diffusivity * self.smoothing_t / self._mass_in
        ).unsqueeze(-1)   # (N, 1)
        std_out = torch.sqrt(
            2.0 * self._diffusivity * self.smoothing_t / self._mass_out
        ).unsqueeze(-1)   # (N, 1)

        _, qphi_base, _ = _base_push_cache.get_or_compute(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
        )

        qphi_diffused_in  = torch.zeros_like(qphi_base)
        qphi_diffused_out = torch.zeros_like(qphi_base)
        for _ in range(self.n_diffusion_samples):
            Z = torch.randn(N, d, device=device, dtype=dtype)

            coords_in = tgt_coords + std_in * Z
            coords_in = coords_in - torch.floor(coords_in)
            _, shifted, _ = _base_push_cache.get_or_compute(
                neural_isometry, initial_dictionary, coords_in, indices, pushforward_kwargs
            )
            qphi_diffused_in = qphi_diffused_in + shifted

            coords_out = tgt_coords + std_out * Z
            coords_out = coords_out - torch.floor(coords_out)
            _, shifted, _ = _base_push_cache.get_or_compute(
                neural_isometry, initial_dictionary, coords_out, indices, pushforward_kwargs
            )
            qphi_diffused_out = qphi_diffused_out + shifted

        qphi_diffused_in  = qphi_diffused_in  / self.n_diffusion_samples
        qphi_diffused_out = qphi_diffused_out / self.n_diffusion_samples

        log_mass_in  = torch.log(self._mass_in)
        log_mass_out = torch.log(self._mass_out)
        qform_in  = parallel_inner_product(qphi_base, qphi_diffused_in,  logabsdet=log_mass_in)
        qform_out = parallel_inner_product(qphi_base, qphi_diffused_out, logabsdet=log_mass_out)

        ch0 = (indices[:, -1] == 0)  # (A,)
        return torch.where(ch0, -qform_in, -qform_out)
