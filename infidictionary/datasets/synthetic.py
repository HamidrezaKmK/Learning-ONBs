import torch
import math

from abc import ABC, abstractmethod

from infidictionary.datasets.base import IrregularDataset
from infidictionary.domain_samplers import DomainSampler, LineSegmentSampler


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
