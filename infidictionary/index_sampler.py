from typing import Dict, List

import torch
from abc import ABC, abstractmethod

class IndexSampler(ABC):
    @abstractmethod
    def sample_indices(self, num_samples: int, i_epoch: int, n_epoch: int):
        raise NotImplementedError

class UniformIndexSampler(IndexSampler):
    def __init__(self, n_indices: int):
        self.n_indices = n_indices

    def sample_indices(self, num_samples: int, i_epoch: int, n_epoch: int):
        return torch.randint(0, self.n_indices, (num_samples,)).long().flatten()

class GeometricIndexSampler(IndexSampler):
    def __init__(self, base_p: float, final_p: float, gamma: float = 1.0):
        self.base_p = base_p
        self.gamma = gamma
        self.final_p = final_p

    def sample_indices(self, num_samples: int, i_epoch: int, n_epoch: int):
        # do a gamma transformation to adjust p over epochs
        scaled = (i_epoch / n_epoch) ** self.gamma
        p = self.base_p * (1 - scaled) + self.final_p * scaled
        # sample num_samples from a geometric distribution with parameter p
        samples = torch.distributions.Geometric(probs=p).sample((num_samples,))  # shift to start from 0
        return samples.long().flatten()

class HybridSampler(IndexSampler):
    def __init__(
        self,
        index_samplers: Dict[str, IndexSampler],
        weights: Dict[str, float],
    ):
        super().__init__()
        self.index_samplers = []
        self.index_sampler_weights = []
        for key in index_samplers.keys():
            if key not in weights:
                raise ValueError(f"Weight for sampler {key} not provided.")
            self.index_samplers.append(index_samplers[key])
            self.index_sampler_weights.append(weights[key])
        self.index_sampler_weights = torch.tensor(self.index_sampler_weights)
        self.index_sampler_weights = self.index_sampler_weights / self.index_sampler_weights.sum()
    
    def sample_indices(self, num_samples: int, i_epoch: int, n_epoch: int):
        # choose a sampler
        chosen_sampler_idx = torch.multinomial(self.index_sampler_weights, num_samples=1, replacement=False).item()
        chosen_sampler: IndexSampler = self.index_samplers[chosen_sampler_idx]
        return chosen_sampler.sample_indices(num_samples, i_epoch, n_epoch)
