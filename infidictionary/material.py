import math
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Material(ABC):
    """Abstract base class for spatially varying materials.

    The vibrational modes of the material are the eigenfunctions of the
    weighted Laplacian -∇·(D(x) ∇), where D(x) > 0 is the diffusivity
    returned by ``__call__``.

    ``__call__`` returns a ``(diffusivity, mass)`` pair:
      * ``diffusivity`` (N,) — the spatially varying diffusivity D(x).
      * ``mass``        (N,) — the integration weight ρ(x) for the
                               mass-corrected eigenvalue problem Lφ = λ ρ φ.

    Subclasses must implement ``project_to_domain`` to fold coordinates back
    onto the material's domain (e.g. torus wrapping, reflection, clipping).
    """

    @abstractmethod
    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def project_to_domain(self, coords: torch.Tensor) -> torch.Tensor:
        """Project coordinates onto the material's domain.

        Args:
            coords: (N, d) coordinates, possibly outside the canonical domain.

        Returns:
            (N, d) coordinates folded back to the canonical domain.
        """
        raise NotImplementedError

    # ── Derived geometric utilities ────────────────────────────────────────────

    def neighbourhood(
        self,
        coords: torch.Tensor,
        K: int,
        h: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return K symmetric neighbour-pairs around each coordinate.

        For each of the N input coordinates, K random unit directions
        δ_k ~ Uniform(S^{d-1}) are drawn and the pairs

            x⁺_k = project_to_domain(x + h δ_k)
            x⁻_k = project_to_domain(x − h δ_k)

        are returned.  The shared directions are also returned so that callers
        can reconstruct finite-difference gradients as
        ``(f(x⁺) − f(x⁻)) / (2h)``.

        Args:
            coords: (N, d) base coordinates (already in the canonical domain).
            K:      Number of neighbour pairs per point.
            h:      Step size for the perturbation.

        Returns:
            coords_plus:  (N, K, d) perturbed coordinates x + h δ.
            coords_minus: (N, K, d) perturbed coordinates x − h δ.
            directions:   (N, K, d) unit directions δ.
        """
        N, d = coords.shape
        device, dtype = coords.device, coords.dtype

        directions = torch.randn(N, K, d, device=device, dtype=dtype)
        directions = directions / directions.norm(dim=-1, keepdim=True)

        coords_exp = coords.unsqueeze(1)  # (N, 1, d)
        coords_plus  = self.project_to_domain(coords_exp + h * directions)
        coords_minus = self.project_to_domain(coords_exp - h * directions)

        return coords_plus, coords_minus, directions

    def diffuse(
        self,
        coords: torch.Tensor,
        K: int,
        num_steps: int,
        dt: float,
        h_fd: float = 1e-3,
    ) -> torch.Tensor:
        """Run K independent Euler–Maruyama trajectories from each coordinate.

        Simulates the Itô SDE

            dX = (∇D(X) / ρ(X)) dt + √(2 D(X) / ρ(X)) dW

        where ∇D is estimated with one random-direction centered FD step using
        ``neighbourhood``.  After each step coordinates are projected back onto
        the domain via ``project_to_domain``.

        Args:
            coords:    (N, d) starting coordinates.
            K:         Number of independent trajectories per starting point.
            num_steps: Number of Euler–Maruyama steps.
            dt:        Time step size.
            h_fd:      Finite-difference step size for the drift gradient.

        Returns:
            (N, K, d) terminal coordinates after ``num_steps`` steps.
        """
        N, d = coords.shape
        device, dtype = coords.device, coords.dtype

        # Replicate each starting point K times: (N*K, d)
        x = coords.unsqueeze(1).expand(N, K, d).reshape(N * K, d).clone()

        for _ in range(num_steps):
            # Evaluate D(x) and ρ(x) at current positions
            D, rho = self(x)  # each (N*K,)

            # Estimate ∇D(x) via single random-direction centered FD
            delta = torch.randn(N * K, d, device=device, dtype=dtype)
            delta = delta / delta.norm(dim=-1, keepdim=True)
            x_plus  = self.project_to_domain(x + h_fd * delta)
            x_minus = self.project_to_domain(x - h_fd * delta)
            D_plus,  _ = self(x_plus)
            D_minus, _ = self(x_minus)
            # ∇D · δ ≈ (D(x+hδ) − D(x−hδ)) / (2h)  →  ∇D ≈ d * grad_est * δ
            grad_est = (D_plus - D_minus) / (2.0 * h_fd)  # (N*K,)
            grad_D = d * grad_est.unsqueeze(-1) * delta    # (N*K, d)

            # Itô drift: ∇D / ρ
            drift = grad_D / rho.unsqueeze(-1)

            # Diffusion coefficient: √(2 D / ρ)
            diff_coef = torch.sqrt(2.0 * D / rho).unsqueeze(-1)  # (N*K, 1)

            noise = torch.randn_like(x)
            x = self.project_to_domain(x + drift * dt + diff_coef * math.sqrt(dt) * noise)

        return x.reshape(N, K, d)


# ── Concrete materials ─────────────────────────────────────────────────────────


class FourierTorusMaterial(Material):
    """Torus material with a single Fourier-mode diffusivity pattern.

    The domain is the D-dimensional flat torus [0, 1]^D.  The diffusivity is
    a single trigonometric mode:

        D(x) = base_diffusivity
               + amplitude · cos(2π Σᵢ kᵢ (xᵢ − phaseᵢ) + directionality)

    The ``phaseᵢ`` parameters shift the cosine peak along axis i in torus
    units (so ``phaseᵢ = 0.5`` centres the peak at the midpoint of axis i).
    ``directionality`` (in radians) controls the overall sin/cos character:
    at 0 the pattern is a pure cosine, at π/2 a pure (negative) sine.

    Setting all ``frequencies`` to 0 gives a spatially uniform (constant)
    diffusivity equal to ``base_diffusivity``.

    Coordinates outside [0, 1]^D are wrapped periodically.  The mass is
    uniform: ρ(x) = 1 everywhere.

    Args:
        frequencies:      List of integer frequencies [k₁, …, kD], one per
                          spatial dimension.
        phases:           List of spatial peak positions [φ₁, …, φD] in
                          [0, 1] torus units, one per dimension.
        directionality:   Global phase θ in radians (default 0).
        base_diffusivity: Mean diffusivity D₀ > 0.
        amplitude:        Amplitude of the cosine modulation.  For D(x) > 0
                          everywhere one needs |amplitude| < base_diffusivity.
    """

    def __init__(
        self,
        frequencies: list[float],
        phases: list[float],
        directionality: float = 0.0,
        base_diffusivity: float = 1.0,
        amplitude: float = 0.5,
    ):
        if len(frequencies) != len(phases):
            raise ValueError(
                f"frequencies and phases must have the same length, "
                f"got {len(frequencies)} and {len(phases)}."
            )
        self.frequencies = list(frequencies)
        self.phases = list(phases)
        self.directionality = directionality
        self.base_diffusivity = base_diffusivity
        self.amplitude = amplitude

    def project_to_domain(self, coords: torch.Tensor) -> torch.Tensor:
        return coords % 1.0

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = self.project_to_domain(coords)

        freq  = torch.tensor(self.frequencies, dtype=coords.dtype, device=coords.device)
        phase = torch.tensor(self.phases,      dtype=coords.dtype, device=coords.device)

        # arg_n = 2π Σᵢ kᵢ (xᵢ − φᵢ) + θ
        arg = 2.0 * math.pi * ((coords - phase) @ freq) + self.directionality

        diffusivity = self.base_diffusivity + self.amplitude * torch.cos(arg)
        diffusivity = diffusivity.clamp(min=1e-6)

        mass = torch.ones_like(diffusivity)
        return diffusivity, mass


class JosephFourierMaterial(Material):
    """Material whose diffusivity is derived from the Joseph Fourier portrait.

    At initialisation the image is:
      1. Optionally thresholded to a binary mask; pixels inside/outside the
         threshold are assigned ``interior_diffusivity`` / ``exterior_diffusivity``
         respectively.
      2. Optionally smoothed with a Gaussian filter (``sigma`` in [0, 1]
         coordinates, converted to pixels internally).

    At query time the stored diffusivity map is sampled with nearest-neighbour
    interpolation — the smoothing is entirely pre-baked into ``self._image``.

    Coordinate convention:
        coords[:, 0] = x ∈ [0, 1]  — horizontal (left → right)
        coords[:, 1] = y ∈ [0, 1]  — vertical   (bottom → top)

    Coordinates outside [0, 1]² are wrapped periodically before lookup.

    Args:
        image_path:            Path to the Joseph Fourier portrait image.
        image_size:            Resize the image to (image_size × image_size) pixels.
        interior_diffusivity:  Diffusivity assigned to pixel value 0 (darkest /
                               below threshold).
        exterior_diffusivity:  Diffusivity assigned to pixel value 1 (brightest /
                               above threshold).
        threshold:             If given, defines the inside/outside boundary for
                               the mass map: pixels ≤ threshold → inside (ρ = 1),
                               pixels > threshold → outside (ρ = exterior_mass).
                               The diffusivity is always the continuous pixel
                               intensity mapped to the two diffusivity values.
        sigma:                 Gaussian smoothing standard deviation in [0, 1]
                               world coordinates.  0 disables smoothing.
        exterior_mass:         ρ value for pixels outside the material.
                               1.0 = no mass correction.
    """

    def __init__(
        self,
        image_path: str,
        image_size: int = 256,
        interior_diffusivity: float = 0.1,
        exterior_diffusivity: float = 1.0,
        invert: bool = False,
        threshold: float | None = None,
        sigma: float = 0.0,
        exterior_mass: float = 1.0,
    ):
        from scipy.ndimage import gaussian_filter

        img = Image.open(image_path).convert("L")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W) in [0, 1]

        # Flip vertically so y=0 → bottom, y=1 → top.
        img_np = np.flipud(img_np).copy()

        # Mass: threshold determines inside/outside; diffusivity stays continuous.
        if threshold is not None:
            inside = img_np <= threshold  # True = inside the material
        else:
            inside = np.ones_like(img_np, dtype=bool)

        self.exterior_mass = exterior_mass
        mass_np = np.where(inside, 1.0, exterior_mass).astype(np.float32)
        self._mass_image: torch.Tensor = (
            torch.from_numpy(mass_np).unsqueeze(0).unsqueeze(0)
        )

        # Diffusivity: map raw pixel intensity → [interior, exterior].
        img_np = interior_diffusivity + (exterior_diffusivity - interior_diffusivity) * img_np
        if invert:
            img_np = exterior_diffusivity + interior_diffusivity - img_np

        # Gaussian smoothing in pixel space (sigma in world coords → pixels).
        if sigma > 0.0:
            img_np = gaussian_filter(img_np, sigma=sigma * image_size)
            img_np = np.clip(img_np, interior_diffusivity, exterior_diffusivity)

        # (1, 1, H, W) for F.grid_sample
        self._image: torch.Tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)

    def project_to_domain(self, coords: torch.Tensor) -> torch.Tensor:
        return coords % 1.0

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbour lookup of the pre-smoothed diffusivity and mass maps.

        Args:
            coords: (N, 2) coordinates; wrapped periodically to [0, 1]².

        Returns:
            diffusivity: (N,) in [interior_diffusivity, exterior_diffusivity].
            mass:        (N,) — 1.0 inside, exterior_mass outside.
        """
        coords = self.project_to_domain(coords)

        # F.grid_sample expects grid in [-1, 1]² with (x, y) ordering,
        # shape (1, 1, N, 2) for point queries.
        grid = (coords * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        img = self._image.to(coords.device, coords.dtype)
        mass_img = self._mass_image.to(coords.device, coords.dtype)

        sample_kwargs = dict(mode="nearest", padding_mode="border", align_corners=True)
        diffusivity = F.grid_sample(img, grid, **sample_kwargs).reshape(-1)
        mass = F.grid_sample(mass_img, grid, **sample_kwargs).reshape(-1)
        return diffusivity, mass


class AirplaneMaterial(Material):
    """Material defined on the silhouette of an airplane.

    The diffusivity is higher inside the airplane structure (body/wing/tail)
    and lower in the background, creating vibrational modes that capture the
    airplane's shape.

    Coordinates outside [0, 1]² are wrapped periodically before evaluation.

    Args:
        interior_diffusivity: Diffusivity of the airplane structure.
        exterior_diffusivity: Diffusivity of the background.
        length:               Controls the proportions of the airplane shape.
        exterior_mass:        ρ value for background points outside the airplane.
                              1.0 = no mass correction.
    """

    def __init__(
        self,
        interior_diffusivity: float = 1.0,
        exterior_diffusivity: float = 0.1,
        length: float = 2.0,
        exterior_mass: float = 1.0,
    ):
        super().__init__()
        self.interior_diffusivity = interior_diffusivity
        self.exterior_diffusivity = exterior_diffusivity
        self.length = length
        self.exterior_mass = exterior_mass

    def project_to_domain(self, coords: torch.Tensor) -> torch.Tensor:
        return coords % 1.0

    def _get_body(self, coords: torch.Tensor) -> torch.Tensor:
        t = (coords.clone() - 0.5) * 2.0

        t[:, 1] = t[:, 1] / max(self.length, 3)

        idx_pos = t[:, 1] > 0
        t[idx_pos, 0] = t[idx_pos, 0] * (1.2 + (self.length - 1.6) * 0.35)

        idx_neg = t[:, 1] < 0
        t[:, 0] = t[:, 0] * (0.25 + t[:, 1].abs()) * (4 - self.length) * 1.5
        t[idx_neg, 0] = t[idx_neg, 0] / (1 / 1.5 + (self.length - 1.6) * 0.4)

        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        r, l = 0.05, 0.125

        mask[(t[:, 0] ** 2 + (t[:, 1] - l) ** 2) < r * r] = True
        mid = t[:, 0].abs() < r
        mid &= t[:, 1].abs() < l
        mask[mid] = True
        mask[(t[:, 0] ** 2 + (t[:, 1] + l) ** 2) < r * r] = True

        return mask

    def _get_wing(self, coords: torch.Tensor) -> torch.Tensor:
        t = (coords.clone() - 0.5) * 2.0

        t[:, 1] = t[:, 1] * 8 / 4.5 / self.length * 2.5
        t[:, 1] = t[:, 1] + 0.65 - 0.275 * self.length
        t[:, 1] = t[:, 1] + (-2.8 + self.length * 1.5) * t[:, 0].abs() * (self.length - 1.6) / (2.6 - 1.6)
        t[:, 0] = t[:, 0] * (self.length + 3) / 10

        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        mask[t[:, 1] + (0.05 + self.length * 0.35) * (t[:, 0].abs() * (self.length - 1.6) / (2.6 - 1.6)) < 0.45] = True
        mask[t[:, 1] < t[:, 0].abs() * 0.5 + 0.55 - (self.length - 0.25) * 0.45] = False
        mask[t[:, 0].abs() > 0.4 - 0.175 * (self.length - 1.6)] = False

        return mask

    def _get_tail(self, coords: torch.Tensor) -> torch.Tensor:
        t = (coords.clone() - 0.5) * 2.0

        t[:, 1] = t[:, 1] + max(self.length, 3) * 0.16
        t[:, 1] = t[:, 1] * 8 / 4 * 2 / (1 + (self.length - 1.6) * 0.4)
        t[:, 1] = t[:, 1] + 0.15
        t[:, 1] = t[:, 1] + 0.7 * (self.length - 1.6) / (2.6 - 1.6) * t[:, 0].abs()
        t[:, 0] = t[:, 0] * 5 / 10 * 4 / (1 + (self.length - 1.6) * 0.8)

        mask = torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
        mask[t[:, 1] + 0.85 * t[:, 0].abs() < 0.35] = True
        mask[t[:, 1] < 0] = False
        mask[t[:, 0].abs() > 0.25] = False

        return mask

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return diffusivity and mass.

        Inside the airplane (body/wing/tail): interior_diffusivity, mass = 1.
        Outside (background): exterior_diffusivity, mass = exterior_mass.
        """
        coords = self.project_to_domain(coords)

        body_mask = self._get_body(coords)
        wing_mask = self._get_wing(coords)
        tail_mask = self._get_tail(coords)
        inside = body_mask | wing_mask | tail_mask

        diffusivity = torch.full(
            (coords.shape[0],), self.exterior_diffusivity,
            device=coords.device, dtype=torch.float32,
        )
        diffusivity[inside] = self.interior_diffusivity

        mass = torch.ones(coords.shape[0], device=coords.device, dtype=torch.float32)
        mass[~inside] = self.exterior_mass

        return diffusivity, mass
