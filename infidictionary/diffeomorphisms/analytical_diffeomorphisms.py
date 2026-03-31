"""
Analytical (parameter-free) diffeomorphisms between the unit square and the unit disk.

These are closed-form maps that convert between ``[0, 1]^2`` and the unit disk in ℝ².
Because they have no learnable parameters, they serve as fixed coordinate-change
building blocks — primarily used by ``DiskFlow`` (in ``disk_flows.py``) to conjugate
a square-based normalizing flow into a disk-based one.

Overview
--------
``PolarTransform``
    Area-preserving map ``[0, 1]^2 ↔ disk`` via a square-root radius and linear angle.
    Simple and fast but introduces mild angular distortion near the boundary.

``ShirlyChiu``
    Low-distortion map ``[0, 1]^2 ↔ disk`` via the Shirley–Chiu concentric mapping.
    Sectors of the square map to equal-area sectors of the disk, reducing distortion
    compared to the polar approach.
"""

from .base import Diffeomorphism
import torch
import math


class PolarTransform(Diffeomorphism):
    """Area-preserving diffeomorphism between ``[0, 1]^2`` and the unit disk.

    The mapping uses a square-root radius to preserve area element:

    * ``u ∈ [0, 1]`` → radius ``r = √u``  (so that ``r²`` is uniform, preserving area)
    * ``v ∈ [0, 1]`` → angle ``θ = 2πv - π``  (linearly mapped to ``[-π, π]``)
    * Cartesian output: ``(x, y) = (r cos θ, r sin θ)``

    The Jacobian determinant of this map is the constant ``π``, so
    ``logabsdet = log π`` for :meth:`forward` and ``-log π`` for :meth:`inverse`.
    """

    def __init__(self):
        super().__init__()
        self.pi = torch.tensor(math.pi)
        self.log_pi = torch.tensor(math.log(math.pi))

    def forward(self, u_square: torch.Tensor):
        """Map from the unit square ``[0, 1]^2`` to the unit disk.

        Args:
            u_square: Tensor of shape ``(N, 2)`` with values in ``[0, 1]``.
                Column 0 is the radius-like variable; column 1 is the angle-like variable.

        Returns:
            Tuple ``(xy, logabsdet)`` where ``xy`` has shape ``(N, 2)`` with points
            inside the unit disk and ``logabsdet`` has shape ``(N,)`` filled with ``log π``.
        """
        # u_square: (N, 2) in [0, 1]
        u = u_square[:, 0] # Radius-like variable
        v = u_square[:, 1] # Angle-like variable

        # 1. Apply Square Root to radius to preserve area
        # r^2 ~ Uniform => r ~ sqrt(Uniform)
        r = torch.sqrt(u)

        # 2. Map v to [-pi, pi] or [0, 2pi]
        theta = 2 * self.pi * v - self.pi

        # 3. Polar to Cartesian
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        xy = torch.stack([x, y], dim=1)
        logabsdet = torch.full((u_square.shape[0],), self.log_pi, device=u_square.device, dtype=u_square.dtype)
        return xy, logabsdet

    def inverse(self, xy_disk: torch.Tensor):
        """Map from the unit disk back to the unit square ``[0, 1]^2``.

        Args:
            xy_disk: Tensor of shape ``(N, 2)`` with Cartesian coordinates inside
                the unit disk.

        Returns:
            Tuple ``(u_square, logabsdet)`` where ``u_square`` has shape ``(N, 2)``
            with values in ``[0, 1]^2`` and ``logabsdet`` has shape ``(N,)`` filled
            with ``-log π``.
        """
        x = xy_disk[:, 0]
        y = xy_disk[:, 1]

        # 1. Cartesian to Polar
        # Note: We compute r_squared directly to avoid sqrt then square
        r_squared = x**2 + y**2
        theta = torch.atan2(y, x) # in [-pi, pi]

        # 2. Inverse of the volume preserving map
        # u = r^2 (No sqrt here! This is the key fix)
        u = r_squared

        # 3. Normalize theta to [0, 1]
        v = (theta + self.pi) / (2 * self.pi)

        u_square = torch.stack([u, v], dim=1)
        logabsdet = torch.full((xy_disk.shape[0],), -self.log_pi, device=xy_disk.device, dtype=xy_disk.dtype)

        return u_square, logabsdet # logabsdet is zero since this is an isometry

class ShirlyChiu(Diffeomorphism):
    """Low-distortion diffeomorphism between ``[0, 1]^2`` and the unit disk.

    Uses the Shirley–Chiu *concentric* mapping, which partitions the canonical square
    ``[-1, 1]^2`` into four sectors (East, North, West, South) and maps each sector to
    the corresponding quarter of the disk while preserving area.  This reduces the
    angular shear that the simpler polar transform introduces near the corners.

    The Jacobian determinant is the constant ``π`` (same as :class:`PolarTransform`),
    so ``logabsdet = log π`` for :meth:`forward` and ``-log π`` for :meth:`inverse`.

    Reference:
        Shirley & Chiu, "A Low Distortion Map Between Disk and Square", 1997.
    """

    def __init__(
        self,
    ):
        """
        Maps [0, 1]^2 -> Unit Disk (radius=1) via Shirley-Chiu.
        """
        super().__init__()
        self.epsilon = torch.tensor(1e-12)
        self.pi = torch.tensor(math.pi)
        self.log_pi = torch.tensor(math.log(math.pi))

    def forward(self, xy: torch.Tensor):
        """Map from the unit square ``[0, 1]^2`` to the unit disk.

        Internally the input is rescaled to the canonical square ``[-1, 1]^2``, then
        the concentric mapping assigns each point to one of four sectors (East, North,
        West, South) and maps it to the corresponding disk sector.

        Args:
            xy: Tensor of shape ``(N, 2)`` with values in ``[0, 1]``.

        Returns:
            Tuple ``(uv, logabsdet)`` where ``uv`` has shape ``(N, 2)`` with Cartesian
            coordinates inside the unit disk and ``logabsdet`` has shape ``(N,)``
            filled with ``log π``.
        """
        x, y = xy[:, 0], xy[:, 1]
        # 1. Map [0, 1] -> [-1, 1] (Canonical Square)
        a = 2 * x - 1
        b = 2 * y - 1

        # 2. Polar coordinates placeholders
        r = torch.zeros_like(a)
        phi = torch.zeros_like(a)

        # 3. Sector Classification
        # We classify based on the lines y=x and y=-x
        # Region 1 (East): a > |b|
        mask_east = (a > torch.abs(b))
        # Region 2 (North): b > |a|
        mask_north = (b > torch.abs(a))
        # Region 3 (West): a < -|b|
        mask_west = (a < -torch.abs(b))
        # Region 4 (South): b < -|a|
        mask_south = (b < -torch.abs(a))

        # 4. Apply Concentric Mapping
        # Region 1 (East)
        if mask_east.any():
            r[mask_east] = a[mask_east]
            # phi = (pi/4) * (b/a)
            phi[mask_east] = (self.pi / 4) * (b[mask_east] / (a[mask_east] + self.epsilon))

        # Region 2 (North)
        if mask_north.any():
            r[mask_north] = b[mask_north]
            # phi = (pi/2) - (pi/4) * (a/b)
            phi[mask_north] = (self.pi / 2) - (self.pi / 4) * (a[mask_north] / (b[mask_north] + self.epsilon))

        # Region 3 (West)
        if mask_west.any():
            r[mask_west] = -a[mask_west]
            # phi = pi + (pi/4) * (b/a)
            phi[mask_west] = self.pi + (self.pi / 4) * (b[mask_west] / (a[mask_west] - self.epsilon))

        # Region 4 (South)
        if mask_south.any():
            r[mask_south] = -b[mask_south]
            # phi = (3pi/2) - (pi/4) * (a/b) ... equivalent to -pi/2 + ...
            phi[mask_south] = (3 * self.pi / 2) - (self.pi / 4) * (a[mask_south] / (b[mask_south] - self.epsilon))

        # 5. Convert to Cartesian on Disk
        u = r * torch.cos(phi)
        v = r * torch.sin(phi)

        return torch.stack([u, v], dim=1), torch.full((u.shape[0],), self.log_pi, device=u.device, dtype=u.dtype)  # logabsdet is zero since this is an isometry

    def inverse(self, uv: torch.Tensor):
        """Map from the unit disk back to the unit square ``[0, 1]^2``.

        Converts Cartesian disk coordinates to polar ``(r, φ)``, classifies the angle
        into one of four sectors, inverts the concentric mapping, then rescales from
        ``[-1, 1]^2`` to ``[0, 1]^2``.

        Args:
            uv: Tensor of shape ``(N, 2)`` with Cartesian coordinates inside the unit disk.

        Returns:
            Tuple ``(xy, logabsdet)`` where ``xy`` has shape ``(N, 2)`` with values in
            ``[0, 1]^2`` and ``logabsdet`` has shape ``(N,)`` filled with ``-log π``.
        """
        u, v = uv[:, 0], uv[:, 1]
        # 1. Cartesian -> Polar
        r = torch.sqrt(u**2 + v**2)
        phi = torch.atan2(v, u) # range [-pi, pi]

        # 2. Canonical Square placeholders
        a = torch.zeros_like(r)
        b = torch.zeros_like(r)

        # Threshold for center (r=0)
        mask_nz = r > self.epsilon

        # 3. Sector Classification (based on angle phi)
        # East: [-pi/4, pi/4]
        mask_east = (phi >= -self.pi/4) & (phi < self.pi/4) & mask_nz
        # North: [pi/4, 3pi/4]
        mask_north = (phi >= self.pi/4) & (phi < 3*self.pi/4) & mask_nz
        # South: [-3pi/4, -pi/4]
        mask_south = (phi >= -3*self.pi/4) & (phi < -self.pi/4) & mask_nz
        # West: The rest (angles > 3pi/4 or < -3pi/4)
        mask_west = mask_nz & ~(mask_east | mask_north | mask_south)

        # 4. Inverse Mapping
        # East
        if mask_east.any():
            # a = r, b = a * (phi / (pi/4))
            a[mask_east] = r[mask_east]
            b[mask_east] = r[mask_east] * phi[mask_east] * (4 / self.pi)

        # North
        if mask_north.any():
            # b = r, a = -b * (phi - pi/2) / (pi/4)
            b[mask_north] = r[mask_north]
            a[mask_north] = -r[mask_north] * (phi[mask_north] - self.pi/2) * (4 / self.pi)

        # South
        if mask_south.any():
            # b = -r, a = b * (phi + pi/2) / (pi/4)  <- careful with sign
            # Correct: a = r * (phi + pi/2) * (4/pi)
            b[mask_south] = -r[mask_south]
            a[mask_south] = r[mask_south] * (phi[mask_south] + self.pi/2) * (4 / self.pi)

        # West
        if mask_west.any():
            a[mask_west] = -r[mask_west]
            # Wrap phi so it is continuous across pi/-pi boundary relative to West axis
            # If phi > 0, dist is phi - pi. If phi < 0, dist is phi + pi.
            phi_w = phi[mask_west]
            phi_adj = torch.where(phi_w > 0, phi_w - self.pi, phi_w + self.pi)
            b[mask_west] = -r[mask_west] * phi_adj * (4 / self.pi)

        # 5. Map [-1, 1] -> [0, 1]
        x = (a + 1) / 2
        y = (b + 1) / 2
        return torch.stack([x, y], dim=1), torch.full((u.shape[0],), -self.log_pi, device=u.device, dtype=u.dtype) # logabsdet is zero since this is an isometry
