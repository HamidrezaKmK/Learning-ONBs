import numpy as np
import torch
import math
from typing import List

from abc import ABC, abstractmethod

from infidictionary.datasets.base import IrregularDataset
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.diffeomorphisms import Diffeomorphism
from infidictionary.domain_samplers import DomainSampler, SquareSampler, DiskSampler, LineSegmentSampler


class FunctionClassGenerator(IrregularDataset, ABC):

    def __init__(
        self,
        domain_sampler: DomainSampler,
        domain_sample_size: int,
        n_functions: int = 1000,
    ):
        self.domain_sampler = domain_sampler
        self.domain_sample_size = domain_sample_size
        self.n_functions = n_functions

    @abstractmethod
    def _eval(self, coords: torch.Tensor, seed: int) -> torch.Tensor:
        raise NotImplementedError("FunctionClassGenerator is an abstract base class.")

    def __call__(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample coordinates and evaluate the function at the given seed.

        Returns:
            coords: (N, d) sampled coordinates
            vals:   (N, C) function values at those coordinates
        """
        coords = self.domain_sampler.sample(self.domain_sample_size)
        vals = self._eval(coords, seed)
        return coords, vals

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        seeds = torch.randint(0, self.n_functions, (batch_size,))
        coords = self.domain_sampler.sample(self.domain_sample_size)
        all_vals = [self._eval(coords, seed.item()) for seed in seeds]
        return coords, torch.stack(all_vals, dim=0)


class RandomBandpassGenerator(FunctionClassGenerator):
    def __init__(
        self,
        domain_sample_size: int,
        l_sin: int,
        l_cos: int,
        n_functions: int = 1000,
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
        domain_sampler = DiskSampler() if radial else SquareSampler()
        super().__init__(domain_sampler, domain_sample_size, n_functions)
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

    def _eval(self, xy: torch.Tensor, seed: int):
        ret = self._call(xy, seed)
        if self.mean_seed is not None:
            mean_val = self._call(xy, self.mean_seed)
            ret = ret + mean_val
        return ret.unsqueeze(-1)

class BasisRandomGenerator(FunctionClassGenerator):
    def __init__(
        self,
        domain_sampler: DomainSampler,
        domain_sample_size: int,
        dictionary: InfiDictionary,
        atom_indices: List,
        diffeomorphism: Diffeomorphism | None,
        n_functions: int = 1000,
    ):
        super().__init__(domain_sampler, domain_sample_size, n_functions)
        self.dictionary = dictionary
        self.atom_indices = torch.tensor(atom_indices, dtype=torch.long)
        self.diffeomorphism = diffeomorphism

    def _eval(self, coords: torch.Tensor, seed: int):
        device = coords.device

        rng = torch.Generator().manual_seed(seed)

        n_atoms = len(self.atom_indices)
        weights = torch.randn(size=(n_atoms,), generator=rng) / math.sqrt(n_atoms)
        weights = weights.to(device)

        if self.diffeomorphism:
            coords, _ = self.diffeomorphism.inverse(coords)

        all_atoms = self.dictionary.get_atoms(
            coords,
            self.atom_indices.to(device),
        ) # (n_atoms, N, c)

        # combine atoms with weights
        combined_atoms = torch.sum(weights[:, None, None] * all_atoms, dim=0) # (N, c)
        return combined_atoms


class OneDimDiscontinuousGenerator(FunctionClassGenerator):
    """1D toy class with a jump discontinuity at x = 0.5.

        f(x) = ε · [ sin(2π k x)     + 2x     ]    for x ≤ 0.5
        f(x) = ε · [ sin(2π k (1-x)) - 2(1-x) ]    for x > 0.5

    where k is drawn from a shifted Geometric on {1, 2, 3, ...}
    (P(K=k) = (1-p)^{k-1} · p) and ε ∈ {-1, +1} is an optional random sign
    (off by default, set `random_sign=True` to enable).

    For integer k, sin(πk) = 0, so the two pieces approach ε·1 and -ε·1 at
    x=0.5 — a jump of magnitude 2 that flips direction with ε.
    f(0)=f(1)=0, so on the torus the only discontinuity is at x=0.5.

    Mean function:
      • `random_sign=False` (default) — non-zero mean. The deterministic
        trend 2x ⨁ -2(1-x) and the residual E_k[sin(2π k x)] both leak in.
      • `random_sign=True`             — **zero by construction** because
        E[ε f(x)] = E[ε] E[f(x)] = 0; useful when you want to skip learning
        a mean function in FPCA.

    Args:
        domain_sample_size: number of quadrature points per call.
        p:                  geometric distribution parameter.
        n_functions:        seed range for randomness across calls.
        seed:               global rng seed offset.
        stratified:         passed to LineSegmentSampler.
        add_noise:          passed to LineSegmentSampler.
        random_sign:        if True, multiply each draw by ε ∈ {-1, +1}
                            (i.i.d. coin flip) so the dataset has zero mean.
    """

    def __init__(
        self,
        domain_sample_size: int,
        p: float = 0.5,
        n_functions: int = 1000,
        seed: int = 42,
        stratified: bool = True,
        add_noise: bool = True,
        random_sign: bool = False,
    ):
        domain_sampler = LineSegmentSampler(
            stratified=stratified, add_noise=add_noise, length=1.0,
        )
        super().__init__(domain_sampler, domain_sample_size, n_functions)
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1); got {p}")
        self.p = float(p)
        self.global_seed = int(seed)
        self._log_one_minus_p = math.log(1.0 - self.p)
        self.random_sign = bool(random_sign)

    def _sample_k(self, seed: int) -> int:
        """Draw k ∈ {1, 2, ...} via inverse-CDF on the shifted geometric."""
        rng = torch.Generator().manual_seed(self.global_seed + int(seed))
        # u ~ Uniform(0, 1) (open at 1 to avoid log(0)); take 1 - u to keep it open at 0.
        u = torch.rand((1,), generator=rng).item()
        u = min(max(u, 1e-12), 1.0 - 1e-12)
        # P(K ≤ k) = 1 - (1-p)^k  ⇒  k = ⌈log(1-u) / log(1-p)⌉
        k = int(math.ceil(math.log(1.0 - u) / self._log_one_minus_p))
        return max(1, k)

    def _eval(self, coords: torch.Tensor, seed: int) -> torch.Tensor:
        if coords.dim() != 2 or coords.shape[-1] != 1:
            raise ValueError(f"coords must have shape (N, 1); got {tuple(coords.shape)}")
        k = self._sample_k(seed)
        x = coords.squeeze(-1)                                              # (N,)
        f_left  = torch.sin(2.0 * math.pi * k * x)         + 2.0 * x        # 0 → +1
        f_right = torch.sin(2.0 * math.pi * k * (1.0 - x)) - 2.0 * (1.0 - x)  # -1 → 0
        f = torch.where(x <= 0.5, f_left, f_right)                          # (N,)
        if self.random_sign:
            # Independent fair coin flip from a sign-specific RNG stream
            # (offset by 1_000_003 so it doesn't collide with k's stream).
            sgn_rng = torch.Generator().manual_seed(self.global_seed + int(seed) + 1_000_003)
            eps = 1.0 if torch.rand((1,), generator=sgn_rng).item() < 0.5 else -1.0
            f = eps * f
        return f.unsqueeze(-1)                                              # (N, 1)
