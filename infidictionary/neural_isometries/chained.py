import inspect
from typing import Sequence, Callable

import torch
from torch import nn

from .base import NeuralIsometry


class ChainedIsometry(NeuralIsometry):
    """Composition of several neural isometries applied in sequence.

    Given isometries ``[Q_1, Q_2, ..., Q_n]``, the pushforward is the
    composition ``Q_n ∘ ... ∘ Q_1`` (``Q_1`` applied first), and the pullback
    is the exact inverse ``Q_1^{-1} ∘ ... ∘ Q_n^{-1}`` (reverse order).  Since
    each link preserves the ``L²`` inner product, so does the chain.

    The links may have heterogeneous signatures — e.g. an
    :class:`EulerianIsometry` expects ``rot_start_time``/``rot_end_time`` while a
    :class:`ChannelOrthogonalIsometry` does not.  Extra keyword arguments passed to
    :meth:`pushforward`/:meth:`pullback`/:meth:`shuffle_model_state` are
    forwarded to each link, but **filtered to the arguments that link actually
    accepts**, so a single call drives the whole chain.

    Args:
        isometries: Ordered sequence of :class:`NeuralIsometry` links.
    """

    def __init__(
        self, 
        coords_dim: int, 
        channels_dim: int,
        isometries_partial: Sequence[Callable[..., NeuralIsometry]],
    ):
        super().__init__()

        if len(isometries_partial) == 0:
            raise ValueError("ChainedIsometry requires at least one isometry")
        
        self.isometries = nn.ModuleList(
            [iso(coords_dim=coords_dim, channels_dim=channels_dim) for iso in isometries_partial]
        )

    @staticmethod
    def _filter_kwargs(fn, kwargs: dict) -> dict:
        """Keep only the kwargs that ``fn`` accepts (all of them if it has **kwargs)."""
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return dict(kwargs)
        return {k: v for k, v in kwargs.items() if k in params}

    def pushforward(
        self,
        src_coords: torch.Tensor,    # (N, d)
        src_logabsdet: torch.Tensor, # (N,)
        src_field: torch.Tensor,     # (B, N, C)
        **kwargs,
    ):
        coords, logabsdet, field = src_coords, src_logabsdet, src_field
        for iso in self.isometries:
            coords, logabsdet, field = iso.pushforward(
                coords, logabsdet, field, **self._filter_kwargs(iso.pushforward, kwargs)
            )
        return coords, logabsdet, field

    def pullback(
        self,
        tgt_coords: torch.Tensor,    # (N, d)
        tgt_logabsdet: torch.Tensor, # (N,)
        tgt_field: torch.Tensor,     # (B, N, C)
        **kwargs,
    ):
        coords, logabsdet, field = tgt_coords, tgt_logabsdet, tgt_field
        for iso in reversed(self.isometries):
            coords, logabsdet, field = iso.pullback(
                coords, logabsdet, field, **self._filter_kwargs(iso.pullback, kwargs)
            )
        return coords, logabsdet, field

    def shuffle_model_state(self, num_steps: int | None = None, **kwargs):
        """Refresh every link, forwarding ``num_steps`` only where it applies."""
        if num_steps is not None:
            kwargs["num_steps"] = num_steps
        for iso in self.isometries:
            iso.shuffle_model_state(**self._filter_kwargs(iso.shuffle_model_state, kwargs))
        return self
