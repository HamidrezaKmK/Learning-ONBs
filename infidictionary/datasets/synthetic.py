import numpy as np
import torch
import math
from typing import List

from abc import ABC, abstractmethod

from infidictionary.dictionaries.base import InfiDictionary

class FunctionClassGenerator(ABC):
    
    @abstractmethod
    def __call__(self, coords: torch.Tensor, seed: int):
        raise NotImplementedError("FunctionClassGenerator is an abstract base class.")

    def get_batch(self, coords: torch.Tensor, seeds: torch.Tensor):
        all_vals = []
        for seed in seeds:
            all_vals.append(self.__call__(coords, seed.item()))
        return torch.stack(all_vals, dim=0)

class RandomBandpassGenerator(FunctionClassGenerator):
    def __init__(
        self,
        l_sin: int,
        l_cos: int,
        radial: bool = False,
        omega_lo: float = 5.,
        omega_hi: float = 10.,
        phase_lo: float = 0.0,
        phase_hi: float = 2.0 * math.pi,
        r_lo: float = 0.5,
        r_hi: float = 1.0,
        seed: int = 42,
        mean_seed: int | None = None,
    ):
        self.l_sin = l_sin
        self.l_cos = l_cos
        omega_lo = omega_lo
        self.radial = radial
        self.global_seed = seed
        self.rng_dset = torch.Generator().manual_seed(seed)

        self.omega_sin = torch.rand(size=(self.l_sin, 2), generator=self.rng_dset) * (omega_hi - omega_lo) + omega_lo
        self.omega_cos = torch.rand(size=(self.l_cos, 2), generator=self.rng_dset) * (omega_hi - omega_lo) + omega_lo
        self.phase_sin = torch.rand(size=(self.l_sin,), generator=self.rng_dset) * (phase_hi - phase_lo) + phase_lo
        self.phase_cos = torch.rand(size=(self.l_cos,), generator=self.rng_dset) * (phase_hi - phase_lo) + phase_lo
        self.r_sin = torch.rand(size=(self.l_sin,), generator=self.rng_dset) * (r_hi - r_lo) + r_lo
        self.r_cos = torch.rand(size=(self.l_cos,), generator=self.rng_dset) * (r_hi - r_lo) + r_lo
        self.mean_seed = mean_seed

    def _call(self, xy: torch.Tensor, seed: int):
        device = xy.device

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")

        if self.radial:
            # convert to polar coordinates
            r = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)  # (N,)
            theta = torch.atan2(xy[:, 1], xy[:, 0])  # (N,)
            theta = (theta + math.pi) / (2.0 * math.pi)  # scale theta to [0,1]
            xy = torch.stack([r, theta], dim=1)  # (N, 2)
        
        rng_global = torch.Generator().manual_seed(self.global_seed)
        rng = torch.Generator().manual_seed(
            seed + torch.randint(0, 10000, (1,), generator=rng_global).item()
        )
    
        # evaluate dictionary functions
        inner_product_cos = torch.einsum("ij,kj->ik", self.omega_cos.to(device), xy)  # (l_X, N)
        inner_product_cos = inner_product_cos + self.phase_cos.to(device).unsqueeze(1)  # (l_X, N)
        e_cos = torch.cos(inner_product_cos) * self.r_cos.to(device).unsqueeze(1)  # (l_X, N)
        
        inner_product_sin = torch.einsum("ij,kj->ik", self.omega_sin.to(device), xy)  # (l_Y, N)
        inner_product_sin = inner_product_sin + self.phase_sin.to(device).unsqueeze(1)  # (l_Y, N)
        e_sin = torch.sin(inner_product_sin) * self.r_sin.to(device).unsqueeze(1)  # (l_Y, N)

        all_e = torch.cat([e_cos, e_sin], dim=0)  # (l_X + l_Y, N)

        # sample random weights and combine
        weights = torch.randn(size=(self.l_sin + self.l_cos,), generator=rng) / math.sqrt(self.l_sin + self.l_cos)  # (l_X + l_Y,)
        weights = weights.to(device)

        val = (weights.unsqueeze(1) * all_e).sum(axis=0)  # (N,)
        return val
    
    def __call__(self, xy: torch.Tensor, seed: int):
        ret = self._call(xy, seed)
        if self.mean_seed is not None:
            mean_val = self._call(xy, self.mean_seed)
            ret = ret + mean_val
        return ret

class BasisRandomGenerator(FunctionClassGenerator):
    def __init__(
        self,
        dictionary: InfiDictionary,
    ):
        self.dictionary = dictionary
    
    def __call__(self, xy: torch.Tensor, seed: int):
        device = xy.device

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (N, 2)")
        rng = torch.Generator().manual_seed(seed)

        # sample random weights and combine
        num_dictionary_fns = self.dictionary.num_atoms
        if num_dictionary_fns is None:
            raise ValueError("dictionaryRandomGenerator requires finite dictionary.")
        
        weights = torch.randn(size=(num_dictionary_fns,), generator=rng) / math.sqrt(num_dictionary_fns)
        weights = weights.to(device)
        # return a weighted combination of dictionary functions
        all_e = []
        for idx in range(num_dictionary_fns):
            fn_val = self.dictionary.get_atom(xy, idx)
            all_e.append(fn_val)
        all_e = torch.stack(all_e, dim=0)
        val = (weights.unsqueeze(1) * all_e).sum(axis=0)
        return val
