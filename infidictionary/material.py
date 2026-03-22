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

    Subclasses must implement ``__call__``.
    """

    domain_sampler: DomainSampler

    @abstractmethod
    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Return the diffusivity at the given coordinates.

        Args:
            coords: (N, d) coordinates sampled from the material's domain.

        Returns:
            diffusivity: (N,) positive tensor.
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

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Return a constant diffusivity for every coordinate.

        Args:
            coords: (N, d) coordinates.

        Returns:
            diffusivity: (N,) tensor filled with ``self.diffusivity``.
        """
        return torch.full(
            (coords.shape[0],),
            self.diffusivity,
            dtype=coords.dtype,
            device=coords.device,
        )


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
        threshold:        If given, binarise: pixels > threshold → min_diffusivity,
                          pixels ≤ threshold → max_diffusivity.
        sigma:            Gaussian smoothing standard deviation in [0, 1] world
                          coordinates (converted to ``sigma * image_size`` pixels).
                          0 disables smoothing.
        square_sampler_kwargs: Extra keyword arguments forwarded to ``SquareSampler``.
    """

    def __init__(
        self,
        image_path: str,
        image_size: int = 256,
        min_diffusivity: float = 0.1,
        max_diffusivity: float = 1.0,
        threshold: float | None = None,
        sigma: float = 0.0,
        **square_sampler_kwargs,
    ):
        from scipy.ndimage import gaussian_filter

        img = Image.open(image_path).convert("L")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H, W) in [0, 1]

        # Flip vertically so y=0 → bottom, y=1 → top.
        img_np = np.flipud(img_np).copy()

        # Threshold → binary mask, then scale to [min, max].
        if threshold is not None:
            img_np = np.where(img_np > threshold, 0.0, 1.0)

        img_np = min_diffusivity + (max_diffusivity - min_diffusivity) * img_np

        # Gaussian smoothing in pixel space (sigma in world coords → pixels).
        if sigma > 0.0:
            img_np = gaussian_filter(img_np, sigma=sigma * image_size)
            img_np = np.clip(img_np, min_diffusivity, max_diffusivity)

        # (1, 1, H, W) for F.grid_sample
        self._image: torch.Tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        self.domain_sampler = SquareSampler(**square_sampler_kwargs)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """Nearest-neighbour lookup of the pre-smoothed diffusivity map.

        Args:
            coords: (N, 2) coordinates in [0, 1]².

        Returns:
            diffusivity: (N,) in [min_diffusivity, max_diffusivity].
        """
        # F.grid_sample expects grid in [-1, 1]² with (x, y) ordering,
        # shape (1, 1, N, 2) for point queries.
        grid = (coords * 2.0 - 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        img = self._image.to(coords.device, coords.dtype)
        sampled = F.grid_sample(
            img, grid, mode="nearest", padding_mode="border", align_corners=True
        )  # (1, 1, 1, N)
        return sampled.reshape(-1)  # (N,) already in [min_diffusivity, max_diffusivity]


class AirplaneMaterial(Material):
    """
    Material defined on the silhouette of an airplane, with diffusivity derived 
    from whether the point is in the body, wing, or tail. The diffusivity is higher 
    in the body and lower in the wings and tail, to create interesting vibrational 
    modes that capture the airplane's shape.
    """

    def __init__(
        self,
        min_diffusivity: float = 0.1,
        max_diffusivity: float = 1.0,
        length: float = 2.0,
        **square_sampler_kwargs,
    ):
        super().__init__()
        self.min_diffusivity = min_diffusivity
        self.max_diffusivity = max_diffusivity
        self.length = length
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

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Returns the diffusivity at the given coordinates, with higher diffusivity 
        in the body and lower diffusivity in the wings and tail.
        """
        
        body_mask = self._get_body(coords)
        wing_mask = self._get_wing(coords)
        tail_mask = self._get_tail(coords)
        
        # Initialize the domain with the background diffusivity
        diffusivity = torch.full((coords.shape[0],), self.max_diffusivity, 
                                 device=coords.device, dtype=torch.float32)
        diffusivity[body_mask | wing_mask | tail_mask] = self.min_diffusivity
        
        return diffusivity
