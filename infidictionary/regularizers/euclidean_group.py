from typing import Sequence, Tuple

import torch

from .volume_preserving import VolumePreserving


_QUARTER_TURN_TABLE = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # (cos, sin) for k * 90°


class EuclideanGroup(VolumePreserving):
    """Regularizer enforcing ``Q phi_k(x) ≈ phi_k(T(x))`` for a fixed
    transformation ``T`` of the d-torus.

    For ``T*`` to be an L²([0,1]^d) isometry, ``T`` must be a torus isometry —
    i.e., it must descend to a well-defined map on R^d / Z^d.  This requires
    the rotation matrix ``R`` to have integer entries, so only quarter-turn
    rotations are admissible.  The full group of orientation-preserving torus
    isometries (combined with the separate ``mirrors`` parameter) is the
    dihedral / hyperoctahedral group of signed axis permutations.

    Parameters
    ----------
    rotation_quarter_turns : sequence of int, length ``d``
        For each ``i ∈ {0, …, d-1}``, applies a rotation by
        ``k_i * 90°`` in the plane spanned by axes ``(i, (i+1) mod d)``.
        Composed in order: ``R = R_0 @ R_1 @ ... @ R_{d-1}``.

        - d=2: the two entries both act in the (0,1)-plane (so one of them
          should typically be 0 — they compose redundantly).
        - d=3: the three planes (0,1), (1,2), (2,0) generate the 24-element
          rotation symmetry group of the cube.
    """

    def __init__(
        self,
        domain_sample_size: int,
        translation: Tuple,
        rotation_quarter_turns: Sequence[int],
        mirrors: Tuple,
    ):
        super().__init__(domain_sample_size=domain_sample_size)
        self.translation = torch.tensor(translation)
        self.d = len(translation)

        if len(rotation_quarter_turns) != self.d:
            raise ValueError(
                f"rotation_quarter_turns must have length d={self.d}; "
                f"got length {len(rotation_quarter_turns)}"
            )

        R = torch.eye(self.d)
        for i, k in enumerate(rotation_quarter_turns):
            k_int = int(k)
            if k_int != k:
                raise TypeError(
                    f"rotation_quarter_turns[{i}] = {k!r} must be an integer "
                    f"(±k means ±90°·k rotation; rotations are restricted to "
                    f"the dihedral group so T preserves the torus measure)"
                )
            if self.d == 1:
                if k_int % 4 != 0:
                    raise ValueError(
                        "In 1D, the only valid rotation is the identity "
                        "(rotation_quarter_turns must be ≡ 0 mod 4)."
                    )
                continue
            c, s = _QUARTER_TURN_TABLE[k_int % 4]
            j = (i + 1) % self.d
            R_i = torch.eye(self.d)
            R_i[i, i] = c
            R_i[j, j] = c
            R_i[i, j] = s
            R_i[j, i] = -s
            R = R @ R_i

        # By construction R has integer entries; the assert pins this guarantee.
        assert torch.allclose(R, R.round()), f"internal: R not integer-valued: {R}"
        self.rotation = R
        self.mirrors = mirrors

    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        transformed_coordinates = coords @ self.rotation - self.translation
        for dim, mirror in enumerate(self.mirrors):
            if mirror:
                transformed_coordinates[:, dim] = 1.0 - transformed_coordinates[:, dim]
        return transformed_coordinates % 1.0  # wrap around to [0, 1)
