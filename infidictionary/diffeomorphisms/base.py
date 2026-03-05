"""
Base classes for diffeomorphisms.

All diffeomorphisms in this package follow a common convention:
  - ``forward(x)`` maps coordinates in the *source* domain to the *target* domain
    and returns ``(y, logabsdet)`` where ``logabsdet`` is the log-absolute-determinant
    of the Jacobian, i.e. ``log |det J_forward(x)|``.
  - ``inverse(y)`` maps coordinates back from the target domain to the source domain
    and returns ``(x, logabsdet)`` where ``logabsdet = log |det J_inverse(y)|``.

The log-absolute-determinant convention is consistent with normalizing-flow literature:
a positive logabsdet means the map expands volume; negative means it contracts.
"""

from abc import ABC, abstractmethod
import torch
from torch import nn


class Diffeomorphism(ABC, nn.Module):
    """Abstract base class for smooth, invertible maps (diffeomorphisms).

    Subclasses must implement :meth:`forward` and :meth:`inverse`, both of which
    return a ``(coords, logabsdet)`` pair so that log-determinant contributions can
    be accumulated when composing multiple diffeomorphisms.
    """

    @abstractmethod
    def forward(self, coords: torch.Tensor):
        """Map coordinates from the source domain to the target domain.

        Args:
            coords: Tensor of shape ``(N, d)`` containing source-domain coordinates.

        Returns:
            Tuple ``(y, logabsdet)`` where ``y`` has the same shape as ``coords`` and
            ``logabsdet`` is a tensor of shape ``(N,)`` containing
            ``log |det J_forward(coords)|`` for each point.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def inverse(self, coords: torch.Tensor):
        """Map coordinates from the target domain back to the source domain.

        Args:
            coords: Tensor of shape ``(N, d)`` containing target-domain coordinates.

        Returns:
            Tuple ``(x, logabsdet)`` where ``x`` has the same shape as ``coords`` and
            ``logabsdet`` is a tensor of shape ``(N,)`` containing
            ``log |det J_inverse(coords)|`` for each point.
        """
        raise NotImplementedError("Subclasses must implement this method")


class IdentityFlow(Diffeomorphism):
    """The trivial diffeomorphism that leaves coordinates unchanged.

    Both :meth:`forward` and :meth:`inverse` return the input coordinates unchanged
    with a zero log-absolute-determinant.  Useful as a neutral element when building
    :class:`ChainDiffeomorphism` or as a no-op placeholder.
    """

    def forward(self, x: torch.Tensor):
        return x, torch.zeros(x.shape[0], device=x.device)

    def inverse(self, y: torch.Tensor):
        return y, torch.zeros(y.shape[0], device=y.device)


class ChainDiffeomorphism(Diffeomorphism):
    """Composition of an ordered sequence of diffeomorphisms.

    ``forward`` applies each diffeomorphism in list order and accumulates their
    log-absolute-determinants (i.e. ``logabsdet = sum_i logabsdet_i``).
    ``inverse`` applies them in *reverse* order, again accumulating logabsdets.

    Args:
        diffeomorphisms: Ordered list of :class:`Diffeomorphism` instances.
            The first element is applied first in the forward pass.
    """

    def __init__(self, diffeomorphisms: list[Diffeomorphism]):
        super().__init__()
        self.diffeomorphisms = nn.ModuleList(diffeomorphisms)

    def to(self, *args, **kwargs):
        for diffeo in self.diffeomorphisms:
            diffeo.to(*args, **kwargs)
        return self

    def forward(self, x: torch.Tensor):
        logabsdet_total = torch.zeros(x.shape[0], device=x.device)
        for diffeo in self.diffeomorphisms:
            x, logabsdet = diffeo.forward(x)
            logabsdet_total += logabsdet
        return x, logabsdet_total

    def inverse(self, y: torch.Tensor):
        logabsdet_total = torch.zeros(y.shape[0], device=y.device)
        for diffeo in reversed(self.diffeomorphisms):
            y, logabsdet = diffeo.inverse(y)
            logabsdet_total += logabsdet
        return y, logabsdet_total


class InverseDiffeomorphism(Diffeomorphism):
    """Wraps a diffeomorphism and swaps its forward and inverse directions.

    This is useful for constructing chains that need the inverse of an existing
    diffeomorphism without duplicating logic.  Concretely:

    .. code-block:: python

        inv = InverseDiffeomorphism(f)
        inv.forward(x)  # calls f.inverse(x)
        inv.inverse(y)  # calls f.forward(y)

    Args:
        diffeomorphism: The :class:`Diffeomorphism` whose directions should be swapped.
    """

    def __init__(self, diffeomorphism: Diffeomorphism):
        super().__init__()
        self.diffeomorphism = diffeomorphism

    def forward(self, x: torch.Tensor):
        return self.diffeomorphism.inverse(x)

    def inverse(self, y: torch.Tensor):
        return self.diffeomorphism.forward(y)
