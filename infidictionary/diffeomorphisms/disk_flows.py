"""
Learnable diffeomorphisms on the unit disk.

``DiskFlow`` adapts any ``[0,1]^2``-based ``CubeFlow`` to the disk domain by
conjugating it with an analytical square-to-disk coordinate change
(:class:`~infidictionary.diffeomorphisms.analytical_diffeomorphisms.PolarTransform`
or :class:`~infidictionary.diffeomorphisms.analytical_diffeomorphisms.ShirlyChiu`).
"""

from typing import Literal
import torch

from .base import Diffeomorphism, IdentityFlow, ChainDiffeomorphism, InverseDiffeomorphism
from .normalizing_flows import CubeFlow
from .analytical_diffeomorphisms import PolarTransform, ShirlyChiu


class DiskFlow(Diffeomorphism):
    """Learnable diffeomorphism on the unit disk, built by conjugation.

    A diffeomorphism ``φ: disk → disk`` is constructed by:

    1. **disk → square** — apply the inverse of the chosen square-to-disk map
       (:class:`~infidictionary.diffeomorphisms.analytical_diffeomorphisms.PolarTransform`
       or :class:`~infidictionary.diffeomorphisms.analytical_diffeomorphisms.ShirlyChiu`).
    2. **square → square** — apply a :class:`~infidictionary.diffeomorphisms.normalizing_flows.CubeFlow`
       (e.g. ``UnitCubeNeuralSplineFlow`` or ``UnitSquareKumaraswamy``).
       If ``cubeflow`` is ``None``, the identity is used.
    3. **square → disk** — re-apply the forward square-to-disk map.

    This conjugation transfers the expressive power of ``CubeFlow`` to the disk
    domain while ensuring the output stays inside the unit disk.

    Args:
        cubeflow: A :class:`~infidictionary.diffeomorphisms.normalizing_flows.CubeFlow`
            instance operating on ``[0,1]^2``.  Pass ``None`` for the identity (no
            learned warp).
        method: Which square-to-disk coordinate change to use.
            ``'shirley_chiu'`` (default) has lower angular distortion;
            ``'polar'`` is simpler.
    """

    def __init__(
        self,
        cubeflow: CubeFlow | None = None,
        method: Literal['shirley_chiu', 'polar'] = 'shirley_chiu',
    ):
        super().__init__()
        polar_cls = PolarTransform if method == 'polar' else ShirlyChiu
        self.register_module(
            "_diffeomorphism",
            ChainDiffeomorphism(
                [
                    InverseDiffeomorphism(polar_cls()),
                    cubeflow if cubeflow is not None else IdentityFlow(),
                    polar_cls()
                ]
            )
        )

    def forward(self, x: torch.Tensor):
        """Map disk coordinates through disk → square → square → disk.

        Args:
            x: Tensor of shape ``(N, 2)`` with points inside the unit disk.

        Returns:
            Tuple ``(y, logabsdet)`` where ``y`` is inside the unit disk and
            ``logabsdet`` has shape ``(N,)``.
        """
        return self._diffeomorphism.forward(x)

    def inverse(self, y: torch.Tensor):
        """Invert the disk flow: disk → square → square⁻¹ → disk (in reverse).

        Args:
            y: Tensor of shape ``(N, 2)`` with points inside the unit disk.

        Returns:
            Tuple ``(x, logabsdet)`` where ``x`` is inside the unit disk and
            ``logabsdet`` has shape ``(N,)``.
        """
        return self._diffeomorphism.inverse(y)
