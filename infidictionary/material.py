from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from infidictionary.domain_samplers import DomainSampler, SquareSampler


class Material(ABC):
    """Abstract base class for spatially varying materials.

    Each material carries a ``domain_sampler`` attribute that describes the
    spatial domain on which the material is defined.  The vibrational modes of
    the material are the eigenfunctions of the weighted Laplacian
    -∇·(D(x) ∇), where D(x) > 0 is the diffusivity returned by ``__call__``.

    ``__call__`` returns a ``(diffusivity, mass)`` pair:
      * ``diffusivity`` (N,) — the spatially varying diffusivity D(x).
      * ``mass``        (N,) — the integration weight ρ(x) for the mass-
                               corrected eigenvalue problem  Lφ = λ ρ φ.
                               Points inside the material's domain have ρ = 1;
                               exterior / background points may have ρ ≪ 1.

    Subclasses must implement ``__call__``.
    """

    domain_sampler: DomainSampler

    @abstractmethod
    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the diffusivity and mass at the given coordinates.

        Args:
            coords: (N, d) coordinates sampled from the material's domain.

        Returns:
            diffusivity: (N,) positive tensor — D(x).
            mass:        (N,) positive tensor — ρ(x) for the mass matrix.
        """
        raise NotImplementedError


# ── Concrete materials ─────────────────────────────────────────────────────────


class ConstantMaterial(Material):
    """Homogeneous material with spatially uniform diffusivity.

    The domain is arbitrary and supplied via the ``domain_sampler`` argument,
    making it easy to combine a flat diffusivity with non-trivial geometries
    (e.g. an airplane silhouette):

        ConstantMaterial(diffusivity=1.0, domain_sampler=AirplaneSampler())

    Args:
        diffusivity:    Constant positive diffusivity value.
        domain_sampler: Sampler that draws coordinates from the material domain.
    """

    def __init__(self, diffusivity: float, domain_sampler: DomainSampler):
        self.diffusivity = diffusivity
        self.domain_sampler = domain_sampler

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a constant diffusivity and unit mass for every coordinate.

        Args:
            coords: (N, d) coordinates.

        Returns:
            diffusivity: (N,) tensor filled with ``self.diffusivity``.
            mass:        (N,) tensor of ones.
        """
        diffusivity = torch.full(
            (coords.shape[0],),
            self.diffusivity,
            dtype=coords.dtype,
            device=coords.device,
        )
        mass = torch.ones_like(diffusivity)
        return diffusivity, mass


class JosephFourierMaterial(Material):
    """Material whose diffusivity is derived from the Joseph Fourier portrait.

    At initialisation the image is:
      1. Optionally thresholded to a binary mask and rescaled to
         [min_diffusivity, max_diffusivity].
      2. Optionally smoothed with a Gaussian filter (``sigma`` in [0, 1]
         coordinates, converted to pixels internally).

    At query time the stored diffusivity map is sampled with nearest-neighbour
    interpolation — the smoothing is entirely pre-baked into ``self._image``.

    Coordinate convention:
        coords[:, 0] = x ∈ [0, 1]  — horizontal (left → right)
        coords[:, 1] = y ∈ [0, 1]  — vertical   (bottom → top)

    Args:
        image_path:       Path to the Joseph Fourier portrait image.
        image_size:       Resize the image to (image_size × image_size) pixels.
        min_diffusivity:  Diffusivity assigned to pixel value 0 (darkest / below
                          threshold).
        max_diffusivity:  Diffusivity assigned to pixel value 1 (brightest / above
                          threshold).
        threshold:        If given, defines the inside/outside boundary for the
                          mass map: pixels ≤ threshold → inside (ρ = 1),
                          pixels > threshold → outside (ρ = exterior_mass).
                          The diffusivity is always the continuous pixel intensity
                          mapped to [min_diffusivity, max_diffusivity].
        sigma:            Gaussian smoothing standard deviation in [0, 1] world
                          coordinates (converted to ``sigma * image_size`` pixels).
                          0 disables smoothing.
        exterior_mass:    ρ value assigned to pixels outside the material (i.e.
                          pixels > threshold when threshold is set).  1.0 means
                          no mass correction.
        square_sampler_kwargs: Extra keyword arguments forwarded to ``SquareSampler``.
    """

    def __init__(
        self,
        image_path: str,
        image_size: int = 256,
        min_diffusivity: float = 0.1,
        max_diffusivity: float = 1.0,
        invert: bool = False,
        threshold: float | None = None,
        sigma: float = 0.0,
        exterior_mass: float = 1.0,
        **square_sampler_kwargs,
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

        # Diffusivity: always map raw pixel intensity → [min, max], no thresholding.
        img_np = min_diffusivity + (max_diffusivity - min_diffusivity) * img_np
        if invert:
            img_np = max_diffusivity + min_diffusivity - img_np

        # Gaussian smoothing in pixel space (sigma in world coords → pixels).
        if sigma > 0.0:
            img_np = gaussian_filter(img_np, sigma=sigma * image_size)
            img_np = np.clip(img_np, min_diffusivity, max_diffusivity)

        # (1, 1, H, W) for F.grid_sample
        self._image: torch.Tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        self.domain_sampler = SquareSampler(**square_sampler_kwargs)

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest-neighbour lookup of the pre-smoothed diffusivity and mass maps.

        Args:
            coords: (N, 2) coordinates in [0, 1]².

        Returns:
            diffusivity: (N,) in [min_diffusivity, max_diffusivity].
            mass:        (N,) — 1.0 inside, exterior_mass outside.
        """
        # F.grid_sample expects grid in [-1, 1]² with (x, y) ordering,
        # shape (1, 1, N, 2) for point queries.
        grid = (coords * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        img = self._image.to(coords.device, coords.dtype)
        mass_img = self._mass_image.to(coords.device, coords.dtype)

        sample_kwargs = dict(mode="nearest", padding_mode="border", align_corners=True)
        diffusivity = F.grid_sample(img, grid, **sample_kwargs).reshape(-1)
        mass = F.grid_sample(mass_img, grid, **sample_kwargs).reshape(-1)
        return diffusivity, mass


class FramedMaterial(Material):
    """Wraps a square-domain material with a thin frame of fixed diffusivity.

    The base material's coordinates are linearly shrunk into
    ``[frame_width, 1 - frame_width]²`` so that a border of width
    ``frame_width`` (in [0, 1] world coordinates) is left around the
    original content and filled with ``frame_diffusivity``.

    The mass of the frame is ``exterior_mass``; the interior inherits the
    base material's mass.

    Args:
        base_material:     A ``Material`` whose ``domain_sampler`` is a
                           ``SquareSampler``.  Raises ``ValueError`` otherwise.
        frame_diffusivity: Diffusivity assigned to points inside the frame.
        frame_width:       Width of the frame in [0, 1] world coordinates.
                           Defaults to 0.05 (5 % on each side).
        exterior_mass:     ρ value for frame points.  1.0 = no correction.
    """

    def __init__(
        self,
        base_material: "Material",
        frame_diffusivity: float,
        frame_width: float = 0.05,
        exterior_mass: float = 1.0,
    ):
        if not isinstance(base_material.domain_sampler, SquareSampler):
            raise ValueError(
                "FramedMaterial requires the base material to use a SquareSampler."
            )
        self.base_material = base_material
        self.frame_diffusivity = frame_diffusivity
        self.frame_width = frame_width
        self.exterior_mass = exterior_mass
        self.domain_sampler = base_material.domain_sampler

    def __call__(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return diffusivity and mass: frame value on the border, base material inside.

        Args:
            coords: (N, 2) coordinates in [0, 1]².

        Returns:
            diffusivity: (N,) tensor.
            mass:        (N,) tensor.
        """
        w = self.frame_width
        in_frame = (
            (coords[:, 0] < w)
            | (coords[:, 0] > 1.0 - w)
            | (coords[:, 1] < w)
            | (coords[:, 1] > 1.0 - w)
        )

        # Map interior coords back to [0, 1] for the base material.
        scale = 1.0 - 2.0 * w
        inner_coords = (coords - w) / scale
        inner_coords = inner_coords.clamp(0.0, 1.0)

        diffusivity, mass = self.base_material(inner_coords)
        diffusivity[in_frame] = self.frame_diffusivity
        mass[in_frame] = self.exterior_mass
        return diffusivity, mass


class AirplaneMaterial(Material):
    """
    Material defined on the silhouette of an airplane, with diffusivity derived
    from whether the point is in the body, wing, or tail. The diffusivity is higher
    in the body and lower in the wings and tail, to create interesting vibrational
    modes that capture the airplane's shape.

    Args:
        min_diffusivity: Diffusivity of the airplane structure.
        max_diffusivity: Diffusivity of the background.
        length:          Controls the proportions of the airplane shape.
        exterior_mass:   ρ value for background points outside the airplane.
                         1.0 = no mass correction.
    """

    def __init__(
        self,
        min_diffusivity: float = 0.1,
        max_diffusivity: float = 1.0,
        length: float = 2.0,
        exterior_mass: float = 1.0,
        **square_sampler_kwargs,
    ):
        super().__init__()
        self.min_diffusivity = min_diffusivity
        self.max_diffusivity = max_diffusivity
        self.length = length
        self.exterior_mass = exterior_mass
        self.domain_sampler = SquareSampler(**square_sampler_kwargs)

    def _get_body(self, coords: torch.Tensor) -> torch.Tensor:
        # Shift and scale [0, 1] coordinates to [-1, 1] for internal geometry
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

        Inside the airplane (body/wing/tail): min_diffusivity, mass = 1.
        Outside (background): max_diffusivity, mass = exterior_mass.
        """
        body_mask = self._get_body(coords)
        wing_mask = self._get_wing(coords)
        tail_mask = self._get_tail(coords)
        inside = body_mask | wing_mask | tail_mask

        diffusivity = torch.full(
            (coords.shape[0],), self.min_diffusivity,
            device=coords.device, dtype=torch.float32,
        )
        diffusivity[inside] = self.max_diffusivity

        mass = torch.ones(coords.shape[0], device=coords.device, dtype=torch.float32)
        mass[~inside] = self.exterior_mass

        return diffusivity, mass
