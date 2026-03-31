from .base import Regularizer, _eval_pushed_atoms
import torch


class FouriererRegularizer(Regularizer):
    """Combined vibration + masking regularizer for two-channel Fourier dictionaries.

    For atoms with multiindex [..., 0] (inside-domain atoms):
      - Vibration: maximises the heat-kernel QF on the *inside* domain (\rho_in).
      - Masking:   penalises both channels outside the portrait mass.

    For atoms with multiindex [..., 1] (outside-domain atoms):
      - Vibration: maximises the heat-kernel QF on the *outside* domain
                   (\rho_out = (1 - \rho + ε).clamp(ε)).
      - Masking:   penalises both channels inside the portrait mass.

    Total energy per atom:
        E_a = vibration_energy_a + masking_weight * masking_energy_a

    A channel-equalisation term ``‖Qφ_a[:,0] - Qφ_a[:,1]‖²`` is also included
    to discourage degenerate solutions where the two channels collapse.

    Args:
        domain_sampler:      A ``DomainSampler`` providing quadrature points.
        domain_sample_size:  Passed as n_per_dim to domain_sampler.sample.
        material:            A ``Material`` providing D(x) and \rho(x).
        smoothing_t:         Diffusion time t in P̃_t = e^{-t \rho⁻¹ L}.
        n_diffusion_samples: Number of MC draws for the diffusion estimate.
        masking_weight:      Weight λ for the masking term (0 disables it).
        mass_threshold:      Points with mass >= mass_threshold are *inside*.
        exterior_mass_eps:   Floor added to (1 - mass) for the outside mass.
        diversity_weight:    Weight for the channel-equalisation term.
        vibration_weight:    Weight for the vibration term.
    """

    def __init__(
        self,
        domain_sampler,
        domain_sample_size: int,
        material,
        smoothing_t: float = 1.0,
        n_diffusion_samples: int = 4,
        masking_weight: float = 1.0,
        mass_threshold: float = 0.5,
        exterior_mass_eps: float = 1e-4,
        diversity_weight: float = 1.0,
        vibration_weight: float = 1.0,
    ):
        super().__init__(domain_sampler, domain_sample_size)
        self.material = material
        self.smoothing_t = smoothing_t
        self.n_diffusion_samples = n_diffusion_samples
        self.masking_weight = masking_weight
        self.mass_threshold = mass_threshold
        self.exterior_mass_eps = exterior_mass_eps
        self.diversity_weight = diversity_weight
        self.vibration_weight = vibration_weight
        self._diffusivity: torch.Tensor | None = None
        self._mass_in:     torch.Tensor | None = None
        self._mass_out:    torch.Tensor | None = None
        self._inside:      torch.Tensor | None = None
        self._outside:     torch.Tensor | None = None

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        with torch.no_grad():
            diffusivity, mass = self.material(self._coords)
            self._diffusivity = diffusivity.detach()
            self._mass_in     = mass.detach()
            self._mass_out    = (1.0 - mass + self.exterior_mass_eps).clamp(
                min=self.exterior_mass_eps
            ).detach()
            self._inside  = (mass >= self.mass_threshold).float().detach()
            self._outside = 1.0 - self._inside

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        from infidictionary.utils import norm2, parallel_inner_product

        if self._diffusivity is None:
            raise RuntimeError(
                "FouriererRegularizer.update_coordinates must be called before compute_energy."
            )

        device = indices.device
        tgt_coords = self._coords.to(device)
        N, d = tgt_coords.shape
        dtype  = tgt_coords.dtype

        diffusivity = self._diffusivity.to(device)
        mass_in     = self._mass_in.to(device)
        mass_out    = self._mass_out.to(device)

        std_in  = torch.sqrt(2.0 * diffusivity * self.smoothing_t / mass_in).unsqueeze(-1)   # (N, 1)
        std_out = torch.sqrt(2.0 * diffusivity * self.smoothing_t / mass_out).unsqueeze(-1)  # (N, 1)

        src_coords = self._pulled_back_coords.to(device)
        src_logabsdet = self._pulled_back_logabsdet.to(device)

        # ── Base pushforward (shared by vibration and masking) ─────────────────
        _, qphi_base, _ = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs,
            src_coords=src_coords, src_logabsdet=src_logabsdet,
        )
        total_energy = 0.0
        ch0 = (indices[:, -1] == 0)  # (A,)
        
        # ── Vibration: MC estimate of P̃_t Qφ for inside and outside masses ───
        if self.vibration_weight > 0.0:
            qphi_diffused_in  = torch.zeros_like(qphi_base)
            qphi_diffused_out = torch.zeros_like(qphi_base)
            for _ in range(self.n_diffusion_samples):
                Z = torch.randn(N, d, device=device, dtype=dtype)

                coords_in = tgt_coords + std_in * Z
                coords_in = coords_in - torch.floor(coords_in)
                _, shifted_in, _ = _eval_pushed_atoms(
                    neural_isometry, initial_dictionary, coords_in, indices, pushforward_kwargs
                )
                qphi_diffused_in = qphi_diffused_in + shifted_in

                coords_out = tgt_coords + std_out * Z
                coords_out = coords_out - torch.floor(coords_out)
                _, shifted_out, _ = _eval_pushed_atoms(
                    neural_isometry, initial_dictionary, coords_out, indices, pushforward_kwargs
                )
                qphi_diffused_out = qphi_diffused_out + shifted_out

            qphi_diffused_in  = qphi_diffused_in  / self.n_diffusion_samples
            qphi_diffused_out = qphi_diffused_out / self.n_diffusion_samples

            log_mass_in  = torch.log(mass_in)
            log_mass_out = torch.log(mass_out)
            qform_in  = parallel_inner_product(qphi_base, qphi_diffused_in,  logabsdet=log_mass_in)
            qform_out = parallel_inner_product(qphi_base, qphi_diffused_out, logabsdet=log_mass_out)

            vibration_energy = torch.where(ch0, -qform_in, -qform_out)

            total_energy += self.vibration_weight * vibration_energy
        
        # ── Masking: penalise atoms outside their active region ────────────────
        if self.masking_weight > 0.0:
            inside  = self._inside.to(device)[None, :, None]   # (1, N, 1)
            outside = self._outside.to(device)[None, :, None]

            e_ch0 = norm2(qphi_base * outside) / norm2(torch.ones_like(qphi_base) * outside)
            e_ch1 = norm2(qphi_base * inside)  / norm2(torch.ones_like(qphi_base) * inside)
            masking_energy = torch.where(ch0, e_ch0, e_ch1)

            diff_ch = qphi_base[:, :, 0:1] - qphi_base[:, :, 1:2]  # (A, N, 1)
            masking_energy = masking_energy + self.diversity_weight * norm2(diff_ch)

            total_energy += self.masking_weight * masking_energy

        return total_energy
