from typing import Literal, Optional

import torch
import math
from .base import InfiDictionary
from infidictionary.utils import pairwise_inner_product


# Irrational, pairwise-distinct multipliers for the deterministic per-atom
# rank.  By Weyl's equidistribution theorem, the sum-mod-1 of irrational
# multiples of integer components is uniformly distributed in [0, 1) and
# collision-free across any finite set of integer tuples.  These values
# depend only on the index components (no random state, no torch RNG) so
# atom (k_1, …, k_d, c) gets exactly the same rank in every run.
_TIEBREAK_BASIS = (
    0.7548776662466927,    # √2 − 1
    0.6180339887498949,    # (√5 − 1) / 2     (golden ratio conjugate)
    0.7320508075688772,    # √3 − 1
    0.6457513110645907,    # √5 − 2
    0.3027756377319946,    # √11 − 3
    0.2360679774997896,    # √5 − 2 (offset)
    0.1622776601683795,    # √13 − 3
    0.0827625302982196,    # √17 − 4
)


class FourierDictionary(InfiDictionary):
    """Dictionary of real-valued Fourier (trigonometric) atoms on [0,1]^d.

    Each atom is a product of 1-D cosine or sine functions active in exactly
    one output channel, forming a complete orthonormal basis of L²([0,1]^d, R^C).

    Atoms are indexed by ``(k_1, ..., k_d, c)`` where:

    * ``k_i >= 0`` selects ``√2 · cos(2π k_i x_i)`` (constant 1 for ``k_i=0``).
    * ``k_i <  0`` selects ``√2 · sin(2π |k_i| x_i)``.
    * ``c ∈ {0,...,C-1}`` is the output channel index.

    **PMF modes.**
    Two priors are supported, controlled by ``pmf_mode``:

    * ``"power_law"`` (default) — isotropic L2-based power law (described below).
    * ``"uniform"``  — uniform PMF on a finite truncated support.  All atoms
      with ``‖k‖_∞ ≤ truncation - 1`` get probability ``1 / N_atoms``;
      every other atom has probability 0.  This collapses the dictionary to a
      finite orthonormal basis with no infinite tail, so
      :meth:`get_high_probability_indices` returns every supported atom and
      the MC tail in :meth:`monte_carlo_captured_energy` is always empty.

    **Power-law prior.**
    The per-atom prior is the isotropic (L2-based) power law

        P(k, c) = (1 + ‖k‖₂²)^{-α} / Z_α / C,

    where Z_α = Σ_{k ∈ Z^d} (1 + ‖k‖₂²)^{-α} (truncated to ‖k‖_∞ ≤ m_max).

    This ensures that axis-aligned atoms such as ``(1, 0)`` (‖k‖₂²=1) always
    receive strictly higher probability than diagonal atoms such as ``(1, 1)``
    (‖k‖₂²=2), even though both share the same L∞ shell.  The old L∞-based
    PMF assigned equal weight to all atoms in the same L∞ shell, so the
    tiebreaker alone determined their ordering — producing arbitrary,
    non-isotropic rankings.

    α (= ``steepness``) controls how sharply the prior favours low-frequency
    content.  The sum Z_α converges iff α > d/2, so for d = 2 we recommend
    α ≥ 1.5 (default α = 2).

    With ``tiebreak_eps > 0`` the per-atom PMF is multiplied by a
    *deterministic* factor in ``[1 - tiebreak_eps, 1]`` derived from a fixed
    irrational basis applied to the integer multi-index (no torch RNG, no
    run-to-run variation).  This breaks the remaining ties (atoms that share
    the same ‖k‖₂², e.g. (2, 1) and (1, 2)), providing the **strict ordering**
    that the variational principle requires.

    The tiebreaker perturbation is small (``≤ tiebreak_eps``) and we ignore it
    when sampling indices; the resulting bias on the energy estimator is
    O(tiebreak_eps) which we treat as negligible for ε ≪ 1.

    Args:
        domain_dim:    Spatial dimension d.
        num_channels:  Number of output channels C.
        pmf_mode:      ``"power_law"`` (default) or ``"uniform"`` — see above.
        steepness:     Power-law exponent α in (1+‖k‖₂²)^{-α}.  Larger ⇒
                       stronger low-frequency concentration.  Must satisfy
                       α > d/2 (a warning is printed otherwise).
                       Ignored when ``pmf_mode="uniform"``.
        tiebreak_eps:  Per-atom unique perturbation magnitude in [0, 1).
                       Default 1e-3.  Set 0 to disable tiebreaking.
                       Ignored when ``pmf_mode="uniform"``.
        m_max:         Internal truncation for normaliser / inverse-CDF
                       sampling.  Default 1024 is conservative for any
                       reasonable α ≥ 1.5.
                       Ignored when ``pmf_mode="uniform"``.
        truncation:    Half-bandwidth K for ``pmf_mode="uniform"``: the
                       support is {k : ‖k‖_∞ ≤ K - 1}, giving
                       ``(2K - 1)^d · C`` atoms in total, each with PMF
                       ``1 / N_atoms``.  Required when ``pmf_mode="uniform"``.
    """

    def __init__(
        self,
        domain_dim: int,
        num_channels: int,
        pmf_mode: Literal["power_law", "uniform"] = "power_law",
        steepness: float = 2.0,
        tiebreak_eps: float = 1e-3,
        m_max: int = 1024,
        truncation: Optional[int] = None,
    ):
        super().__init__()
        if pmf_mode not in ("power_law", "uniform"):
            raise ValueError(
                f"pmf_mode must be 'power_law' or 'uniform'; got {pmf_mode!r}"
            )

        self.domain_dim   = domain_dim
        self.num_channels = num_channels
        self.pmf_mode     = pmf_mode
        self.steepness    = float(steepness)
        self.tiebreak_eps = float(tiebreak_eps)
        self.m_max        = int(m_max)
        self.is_orthonormal = True

        if pmf_mode == "power_law":
            if not 0.0 <= self.tiebreak_eps < 1.0:
                raise ValueError(f"tiebreak_eps must be in [0, 1); got {tiebreak_eps}")
            if self.steepness <= 0.5 * domain_dim:
                # Sum of (1+m²)^{-α} over m only converges for α > d/2; we still
                # *truncate* to m_max so it's finite numerically, but the prior
                # is essentially uniform across shells, defeating the purpose.
                print(
                    f"[FourierDictionary] warning: steepness={steepness} ≤ d/2 "
                    f"= {0.5 * domain_dim}; the shell-PMF sum would diverge "
                    "in the limit, the prior is mostly uniform within "
                    f"m_max={m_max}.  Pick a larger steepness for a sharper prior."
                )

            # Pre-compute the isotropic (L2-based) shell weight table.
            # W(m) = Σ_{k: ‖k‖_∞ = m} (1 + ‖k‖₂²)^{-α}  — used for sampling.
            # Z_L2 = Σ_m W(m) — global normalisation constant for get_index_pmfs.
            l2_w          = self._precompute_l2_shell_weights()  # (m_max+1,)
            self._Z_L2    = l2_w.sum().item()
            self._l2_shell_cdf = (l2_w / self._Z_L2).cumsum(0)  # for inverse-CDF
            self.truncation       = None
            self._uniform_indices = None
            self._uniform_n_atoms = 0
            self._uniform_pmf     = 0.0
        else:  # pmf_mode == "uniform"
            if truncation is None or truncation < 1:
                raise ValueError(
                    f"truncation must be a positive int when pmf_mode='uniform'; "
                    f"got {truncation!r}"
                )
            self.truncation       = int(truncation)
            self._uniform_indices = self.get_truncated_indices(self.truncation)
            self._uniform_n_atoms = self._uniform_indices.shape[0]
            self._uniform_pmf     = 1.0 / float(self._uniform_n_atoms)
            self._Z_L2            = float("nan")
            self._l2_shell_cdf    = None

    # ── Shell helpers ──────────────────────────────────────────────────────────

    def _precompute_l2_shell_weights(self) -> torch.Tensor:
        """W(m) = Σ_{k: ‖k‖_∞ = m} (1 + ‖k‖₂²)^{-α} for m = 0, …, m_max.

        For d=1 and d=2 computed exactly; for d≥3 the L∞ shell is too large
        to enumerate so we use the axis-aligned lower bound (1 + m²)^{-α}
        × shell_size(m), which equals the exact value when d=1 and is a good
        approximation for small d.
        """
        d     = self.domain_dim
        alpha = self.steepness
        W     = torch.zeros(self.m_max + 1)
        W[0]  = 1.0  # k = 0: (1 + 0)^{-α} = 1
        for m in range(1, self.m_max + 1):
            if d == 1:
                # Shell m = {-m, +m}: both have ‖k‖₂² = m²
                W[m] = 2.0 * float((1.0 + m * m) ** (-alpha))
            elif d == 2:
                # k1 = ±m, k2 ∈ [-m, m]  (two full rows)
                k2  = torch.arange(-m, m + 1).float()
                w_rows = 2.0 * (1.0 + m * m + k2 * k2).pow(-alpha).sum().item()
                # k2 = ±m, k1 ∈ [-(m-1), m-1]  (two columns, corners already counted)
                k1  = torch.arange(-(m - 1), m).float()
                w_cols = 2.0 * (1.0 + k1 * k1 + m * m).pow(-alpha).sum().item()
                W[m] = w_rows + w_cols
            else:
                # d ≥ 3: approximate with shell_size × (1+m²)^{-α}
                shell_sz = float((2 * m + 1) ** d - (2 * m - 1) ** d)
                W[m] = shell_sz * float((1.0 + m * m) ** (-alpha))
        return W

    def _sample_from_shells_l2(self, shells: torch.Tensor) -> torch.Tensor:
        """Sample one spatial index from L∞ shell(m) with prob ∝ (1+‖k‖₂²)^{-α}.

        Uses rejection sampling: propose uniformly from the L∞ shell, accept
        with probability ((1+m²)/(1+‖k‖₂²))^α ∈ (0,1].  Expected iterations
        ≤ 4 for d=2 (acceptance rate ≥ 25% at large m with α=2).
        """
        S = shells.shape[0]
        d = self.domain_dim
        result = torch.zeros(S, d, dtype=torch.long)
        done   = torch.zeros(S, dtype=torch.bool)

        while not done.all():
            m     = shells
            cands = torch.zeros(S, d, dtype=torch.long)
            for dim in range(d):
                r = torch.rand(S)
                cands[:, dim] = (r * (2 * m.float() + 1)).long() - m
            on_shell = cands.abs().amax(dim=-1) == m

            # Acceptance ratio: max PMF in shell is at axis-aligned atom (‖k‖₂²=m²)
            l2sq  = cands.pow(2).sum(dim=-1).float()
            m2    = m.float().pow(2)
            ratio = torch.where(
                m == 0,
                torch.ones(S),
                ((1.0 + m2) / (1.0 + l2sq)).pow(self.steepness),
            )
            accept = on_shell & (torch.rand(S) < ratio) & ~done
            result[accept] = cands[accept]
            done   = done | accept

        return result

    def _deterministic_rank(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-atom rank in [0, 1).  Reproducible across runs.

        Atom (k_1, …, k_d, c) is encoded as the fractional part of
        ``Σ_i b_i · component_i``  with irrational ``b_i``.  By Weyl's
        equidistribution theorem this is dense in [0, 1) and pairwise
        distinct on any finite set of integer tuples.
        """
        n_components = idx.shape[1]
        if n_components > len(_TIEBREAK_BASIS):
            raise ValueError(
                f"_TIEBREAK_BASIS has {len(_TIEBREAK_BASIS)} entries; "
                f"need ≥ {n_components}.  Add more irrationals."
            )
        weights = torch.tensor(
            _TIEBREAK_BASIS[:n_components], dtype=torch.float64, device=idx.device,
        )
        s = (idx.to(torch.float64) * weights[None, :]).sum(dim=-1)
        return (s - s.floor()).float()    # ∈ [0, 1)

    # ── Core dictionary methods ────────────────────────────────────────────────

    def _get_spatial_atoms(
        self,
        coords: torch.Tensor,       # (N, d)
        spatial_idx: torch.Tensor,  # (A, d) signed frequency indices
    ) -> torch.Tensor:              # (A, N)
        """Compute scalar Fourier atoms phi_k(x)."""
        A, N = spatial_idx.shape[0], coords.shape[0]
        vals = torch.ones((A, N), device=coords.device, dtype=coords.dtype)
        for d in range(self.domain_dim):
            d_idx = spatial_idx[:, d]
            freq  = torch.abs(d_idx).float()
            kt    = 2.0 * math.pi * freq[:, None] * coords[:, d][None, :]  # (A, N)
            cos_c = math.sqrt(2.0) * torch.cos(kt)
            sin_c = math.sqrt(2.0) * torch.sin(kt)
            component = torch.where(
                d_idx[:, None] == 0,
                torch.ones_like(cos_c),
                torch.where(d_idx[:, None] > 0, cos_c, sin_c),
            )
            vals *= component
        return vals  # (A, N)

    def sample_indices(self, num_samples: int) -> torch.Tensor:
        """Sample atom indices according to the active prior.

        * ``pmf_mode="power_law"``: proportional to the isotropic L2 prior
          (tiebreak ignored, O(tiebreak_eps) bias).
        * ``pmf_mode="uniform"``: uniform over the truncated support.
        """
        if self.pmf_mode == "uniform":
            picks = torch.randint(0, self._uniform_n_atoms, (num_samples,))
            return self._uniform_indices[picks].clone()

        u        = torch.rand(num_samples)
        shells   = torch.searchsorted(self._l2_shell_cdf, u).clamp(0, self.m_max)
        spatial  = self._sample_from_shells_l2(shells)
        channels = torch.randint(0, self.num_channels, (num_samples,))
        return torch.cat([spatial, channels.unsqueeze(-1)], dim=-1)

    def get_atoms(
        self,
        coords: torch.Tensor,  # (N, d)
        idx: torch.Tensor,     # (A, d+1)  last col = channel
    ) -> torch.Tensor:         # (A, N, C)
        """Evaluate Fourier atoms at the given coordinates.  Atom (k, c)
        equals phi_k(x) in channel c and zero elsewhere."""
        spatial_idx = idx[:, :-1]
        channel_idx = idx[:, -1]
        C = self.num_channels
        A, N = spatial_idx.shape[0], coords.shape[0]

        phi  = self._get_spatial_atoms(coords, spatial_idx)
        vals = torch.zeros((A, N, C), device=coords.device, dtype=coords.dtype)
        c_idx = channel_idx.clamp(0, C - 1)
        vals.scatter_(2, c_idx[:, None, None].expand(A, N, 1), phi.unsqueeze(-1))
        return vals  # (A, N, C)

    def get_index_pmfs(self, idx: torch.Tensor) -> torch.Tensor:
        """Per-atom PMF under the active prior.

        * ``pmf_mode="power_law"``:

              P(k, c) = (1 + ‖k‖₂²)^{-α} / Z_L2 / C
                        · (1 - ε · rank(k, c))            (if ε > 0)

          Atoms with the same L2 norm (e.g. (2,1) and (1,2)) tie on the base
          PMF; the deterministic tiebreaker (see :meth:`_deterministic_rank`)
          resolves the remaining ties.

        * ``pmf_mode="uniform"``:  ``1 / N_atoms`` for atoms within the
          truncated support, ``0`` for atoms outside it.
        """
        if self.pmf_mode == "uniform":
            spatial_idx = idx[:, :-1]
            channel_idx = idx[:, -1]
            in_support = (
                (spatial_idx.abs().amax(dim=-1) <= self.truncation - 1)
                & (channel_idx >= 0)
                & (channel_idx < self.num_channels)
            )
            out = torch.zeros(idx.shape[0], dtype=torch.float32, device=idx.device)
            out[in_support] = self._uniform_pmf
            return out

        spatial_idx = idx[:, :-1]
        l2sq  = spatial_idx.pow(2).sum(dim=-1).float()
        base  = (1.0 + l2sq).pow(-self.steepness) / self._Z_L2 / self.num_channels
        if self.tiebreak_eps > 0.0:
            base = base * (1.0 - self.tiebreak_eps * self._deterministic_rank(idx))
        return base

    def _compute_M_bound(self, tail_probability: float) -> int:
        """Largest L∞ shell radius m such that every atom in shell m has
        PMF ≥ tail_probability.

        The *smallest* PMF in L∞ shell m belongs to the diagonal atom
        (±m, …, ±m) with ‖k‖₂² = d·m².  We march m outward and stop as
        soon as that lower bound drops below the threshold.
        """
        scale  = (1.0 - self.tiebreak_eps) / self.num_channels / self._Z_L2
        last_m = 0
        for m in range(self.m_max + 1):
            max_l2sq = float(self.domain_dim) * float(m) ** 2   # diagonal atom
            atom_pmf = float((1.0 + max_l2sq) ** (-self.steepness)) * scale
            if atom_pmf < tail_probability:
                return last_m
            last_m = m
        return self.m_max

    def get_high_probability_indices(self, tail_probability: float) -> torch.Tensor:
        """Return all (k, c) multi-indices with prior PMF ≥ tail_probability.

        In ``pmf_mode="uniform"`` the support is finite, so this returns the
        full truncated index set (regardless of ``tail_probability``) and the
        MC tail of :meth:`monte_carlo_captured_energy` is always empty.
        """
        if self.pmf_mode == "uniform":
            return self._uniform_indices.clone()

        M = self._compute_M_bound(tail_probability)
        vals = torch.arange(-M, M + 1)
        grids = torch.meshgrid(*[vals] * self.domain_dim, indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)

        C   = self.num_channels
        A_s = spatial_idx.shape[0]
        channels    = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        idx = torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)

        return idx[self.get_index_pmfs(idx) >= tail_probability]

    def get_truncated_indices(self, num_truncated: int) -> torch.Tensor:
        """Return all (k, c) indices within the L∞ ball of radius num_truncated-1."""
        freq = torch.arange(-num_truncated + 1, num_truncated)
        grids = torch.meshgrid(*[freq] * self.domain_dim, indexing='ij')
        spatial_idx = torch.stack(grids, dim=-1).view(-1, self.domain_dim)

        C   = self.num_channels
        A_s = spatial_idx.shape[0]
        channels    = torch.arange(C).unsqueeze(0).expand(A_s, -1).reshape(-1)
        spatial_rep = spatial_idx.unsqueeze(1).expand(-1, C, -1).reshape(-1, self.domain_dim)
        return torch.cat([spatial_rep, channels.unsqueeze(-1)], dim=-1)
