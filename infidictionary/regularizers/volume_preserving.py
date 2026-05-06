from abc import abstractmethod

import torch

from .base import Regularizer, _eval_pushed_atoms
from infidictionary.utils import norm2
from infidictionary.domain_samplers import SquareSampler


class VolumePreserving(Regularizer):
    """Base class for regularizers that enforce ``Q phi_k(x) ≈ phi_k(T(x))``
    for a fixed volume-preserving (measure-preserving) map ``T`` of the
    d-torus ``[0,1]^d``.

    Any volume-preserving torus map ``T`` induces an isometry of
    ``L^2([0,1]^d)`` via ``f -> f ∘ T``. This class implements the shared
    pipeline that pulls atoms back to ``T(x)``, pushes them through the
    learned isometry, and matches the two.

    Subclasses override only ``transform``.
    """

    def __init__(self, domain_sample_size: int) -> None:
        super().__init__(
            domain_sampler=SquareSampler(stratified=True, add_noise=True),
            domain_sample_size=domain_sample_size,
        )

    @abstractmethod
    def transform(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply the volume-preserving map ``T`` to ``coords``.

        ``coords`` and the returned tensor are both shaped ``(N, d)`` and
        live in ``[0, 1)^d`` (the unit torus).
        """
        ...

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        self._transformed_coords = self.transform(self._coords)

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        tgt_coords = self._coords.to(indices.device)
        tgt_transformed_coords = self._transformed_coords.to(indices.device)
        N, _ = tgt_coords.shape
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        true_transformed = initial_dictionary.get_atoms(
            tgt_transformed_coords, indices
        )  # (A, N, C)

        _, pushforwarded_f, _ = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, tgt_coords, indices, pushforward_kwargs
        )  # (A, N, C)

        diff = pushforwarded_f - true_transformed  # (A, N, C)
        return norm2(
            diff, logabsdet=torch.zeros(N, device=device, dtype=dtype)
        )
