from typing import Tuple

from .base import Regularizer, _eval_pushed_atoms
import torch

from infidictionary.utils import norm2, parallel_inner_product
from infidictionary.domain_samplers import SquareSampler

class EuclideanGroup(Regularizer):
    def __init__(
        self,
        domain_sample_size: int,
        translation: Tuple,
        rotation_skew_symmetric: Tuple,
        mirrors: Tuple,
    ):
        super().__init__(
            domain_sampler=SquareSampler(stratified=True, add_noise=False), 
            domain_sample_size=domain_sample_size,
        )
        # turn the translation into a tensor
        self.translation = torch.tensor(translation)
        self.d = len(translation)
        # create a d x d rotation matrix from the rotation tuple
        skew = torch.zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(i + 1, self.d):
                angle = rotation_skew_symmetric[i * self.d + j - (i + 1) * (i + 2) // 2]
                skew[i, j] = angle
                skew[j, i] = -angle
        self.rotation = torch.matrix_exp(skew)
        # 
        self.mirrors = mirrors

    def update_coordinates(self, neural_isometry, pushforward_kwargs) -> None:
        super().update_coordinates(neural_isometry, pushforward_kwargs)
        
        transformed_coordinates = self._coords @ self.rotation - self.translation
        for dim, mirror in enumerate(self.mirrors):
            if mirror:
                transformed_coordinates[:, dim] = 1.0 - transformed_coordinates[:, dim]
        self._transformed_coords = transformed_coordinates % 1.0  # wrap around to [0, 1)

    def compute_energy(
        self, neural_isometry, initial_dictionary, indices, pushforward_kwargs
    ) -> torch.Tensor:
        # TODO: fix the issues that apprear when things are Lagrangian
        tgt_coords = self._coords.to(indices.device)
        tgt_transformed_coords = self._transformed_coords.to(indices.device)
        N, d = tgt_coords.shape
        device = tgt_coords.device
        dtype = tgt_coords.dtype

        true_transformed = initial_dictionary.get_atoms(tgt_transformed_coords, indices)  # (A, N, C)

        _, pushforwarded_f,  _ = _eval_pushed_atoms(
            neural_isometry, initial_dictionary, tgt_coords,  indices, pushforward_kwargs
        ) # (A, N, C)

        diff = pushforwarded_f - true_transformed  # (A, N, C)
        return norm2(
            diff, logabsdet=torch.zeros(N, device=device, dtype=dtype)
        )
