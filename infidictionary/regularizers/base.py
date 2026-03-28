import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod

class _PushedAtomsCache:
    """Fixed-size buffer cache for _eval_pushed_atoms results.

    Stores up to ``buffer_size`` (tgt_coords, indices, result) entries.  On
    each call to ``get_or_compute`` the buffer is searched linearly for a
    matching entry (``torch.allclose`` on coords, ``torch.equal`` on indices).
    On a miss the result is computed, appended, and the oldest entry is evicted
    when the buffer is full (FIFO).

    A module-level instance is shared by all regularizers so that the
    neural-isometry forward pass for any coordinate set is executed only once
    per training step as long as it fits in the buffer.

    Args:
        buffer_size: Maximum number of (coords, indices, result) entries to keep.
        atol:        Absolute tolerance for ``torch.allclose`` coordinate comparison.
    """

    def __init__(self, buffer_size: int = 8, atol: float = 1e-6) -> None:
        self.buffer_size = buffer_size
        self.atol = atol
        self._buffer: list[tuple[torch.Tensor, torch.Tensor, tuple]] = []

    def _coords_match(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.shape == b.shape and torch.allclose(a, b, atol=self.atol)

    def get_or_compute(
        self,
        neural_isometry,
        initial_dictionary,
        tgt_coords: torch.Tensor,
        indices: torch.Tensor,
        pushforward_kwargs: dict,
    ) -> tuple:
        for cached_tgt, cached_idx, value in self._buffer:
            if self._coords_match(cached_tgt, tgt_coords) and torch.equal(cached_idx, indices):
                return value
        value = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
        )
        if len(self._buffer) >= self.buffer_size:
            self._buffer.pop(0)
        self._buffer.append((tgt_coords.detach().clone(), indices.detach().clone(), value))
        return value

    def reset(self) -> None:
        """Clear all cached entries.

        Must be called after each backward() pass to release the computation
        graphs held by cached tensors before the next forward pass.
        """
        self._buffer.clear()


_base_push_cache = _PushedAtomsCache()


def _eval_pushed_atoms(
    neural_isometry,
    initial_dictionary,
    tgt_coords: torch.Tensor,
    indices: torch.Tensor,
    pushforward_kwargs: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pullback → get atoms → pushforward.

    The pullback is performed without gradient tracking (src_coords are
    detached); gradients flow only through the pushforward, matching the
    convention used throughout the codebase.

    Returns:
        init:    (A, N, C) initial atoms at src_coords.
        pushed:  (A, N, C) pushed-forward atoms.
        tgt_pf:  (N, d) output target coordinates from the pushforward.
    """
    from infidictionary.neural_isometries import EulerianIsometry

    N = tgt_coords.shape[0]
    device = tgt_coords.device
    dtype = tgt_coords.dtype

    if isinstance(neural_isometry, EulerianIsometry):
        src_coords = tgt_coords
        src_logabsdet = torch.zeros(N, device=device, dtype=dtype)
    else:
        with torch.no_grad():
            src_coords, src_logabsdet, _ = neural_isometry.pullback(
                tgt_coords=tgt_coords,
                tgt_logabsdet=torch.zeros(N, device=device, dtype=dtype),
                tgt_field=torch.zeros(1, N, 1, device=device, dtype=dtype),
                **pushforward_kwargs,
            )
        src_coords = src_coords.detach()
        src_logabsdet = src_logabsdet.detach()

    init = initial_dictionary.get_atoms(src_coords, indices)
    tgt_pf, _, pushed = neural_isometry.pushforward(
        src_coords=src_coords,
        src_logabsdet=src_logabsdet,
        src_field=init,
        **pushforward_kwargs,
    )
    return init, pushed, tgt_pf


class Regularizer(ABC):
    def update_coordinates(self, coords: torch.Tensor) -> None:
        """Called once per epoch when target-domain coordinates change.

        Subclasses may override to pre-compute coordinate-dependent quantities
        (e.g. KNN graphs, diffusivity maps).  Default implementation is a no-op.

        Args:
            coords: (N, d) the new spatial coordinates.
        """
        pass

    @abstractmethod
    def compute_energy(
        self,
        neural_isometry,
        initial_dictionary,
        tgt_coords: torch.Tensor,
        indices: torch.Tensor,
        pushforward_kwargs: dict,
    ) -> torch.Tensor:
        """Per-atom energy.

        Args:
            neural_isometry:    The isometry Q being optimised.
            initial_dictionary: Source atom dictionary.
            tgt_coords:         (N, d) target-domain quadrature points.
            indices:            (A, ...) atom multi-indices.
            pushforward_kwargs: Forwarded to pullback / pushforward.

        Returns:
            energy: (A,) per-atom scalar energy.
        """
        pass


class PushforwardRegularizer(Regularizer):
    """Regularizer that evaluates energy on pushed-forward atoms.

    Subclasses implement ``_energy_from_atoms``; the pullback/pushforward
    pipeline is handled once here via ``_base_push_cache``.
    """

    def compute_energy(
        self, neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
    ) -> torch.Tensor:
        init, pushed, tgt_pf = _base_push_cache.get_or_compute(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
        )
        return self._energy_from_atoms(tgt_pf, init, pushed, indices)

    @abstractmethod
    def _energy_from_atoms(
        self,
        tgt_coords: torch.Tensor,
        init: torch.Tensor,
        pushed: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Per-atom energy given initial and pushed-forward atoms.

        Args:
            tgt_coords: (N, d) spatial coordinates.
            init:       (A, N, C) initial atoms at src_coords.
            pushed:     (A, N, C) pushed-forward atoms at tgt_coords.
            indices:    (A, ...) atom multi-indices.

        Returns:
            energy: (A,) per-atom energy.
        """
        pass

