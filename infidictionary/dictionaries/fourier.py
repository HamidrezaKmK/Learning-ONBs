
import torch
import math
from typing import Literal

from .base import InfiDictionary
from .utils import sample_from_disk

def cos1d(k, t):
    return math.sqrt(2.0) * torch.cos(2.0 * math.pi * k * t) if k > 0 else torch.ones_like(t)

def sin1d(k, t):
    return math.sqrt(2.0) * torch.sin(2.0 * math.pi * k * t) if k > 0 else torch.zeros_like(t)

class FourierDictionary(InfiDictionary):
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2))

    def _get_atom_by_args(
        self, 
        xy: torch.Tensor, 
        kx: int, 
        ky: int,
        kind: Literal["cc", "cs", "sc", "ss"] = "cc",
    ):
        if kx < 0 or ky < 0:
            raise ValueError("kx, ky must be >= 0")
        if kind not in ("cc", "cs", "sc", "ss"):
            raise ValueError("kind must be one of {'cc','cs','sc','ss'}")

        x, y = xy[:, 0], xy[:, 1]


        fx_c = cos1d(kx, x); fx_s = sin1d(kx, x)
        fy_c = cos1d(ky, y); fy_s = sin1d(ky, y)

        if kind == "cc":
            vals = fx_c * fy_c
        elif kind == "cs":
            vals = fx_c * fy_s
        elif kind == "sc":
            vals = fx_s * fy_c
        else:
            vals = fx_s * fy_s

        return vals  # shape (N,)


    _KINDS_4 = ("cc", "cs", "sc", "ss")
    _KINDS_KX0 = ("cc", "cs")  # kx=0 => fx is constant, only theta varies
    _KINDS_KY0 = ("cc", "sc")  # ky=0 => ftheta is constant, only r varies

    def idx_to_params(self, idx: int):
        """
        Map a single global index to (kx, ky, kind) in the requested order.

        Ordering by rings n=max(kx,ky):
          n=0: (0,0) once
          n>=1: (0,n),
                (1,n),...,(n,n),
                (n,n-1),...,(n,1),
                (n,0)

        Multiplicities:
          (0,n): 2  kinds (cc,cs)
          (n,0): 2  kinds (cc,sc)
          else : 4  kinds (cc,cs,sc,ss)
        """
        if idx < 0:
            raise ValueError("idx must be >= 0")

        # n=0 special case
        if idx == 0:
            return 0, 0, "cc"

        # For idx>=1, find smallest n>=1 with idx <= 4*n*(n+1)
        # (since total up to ring n is 1 + 4*n*(n+1))
        n = math.ceil((-1.0 + math.sqrt(1.0 + idx)) / 2.0)
        if n < 1:
            n = 1

        # index where ring n starts
        ring_start = 1 + 4 * (n - 1) * n  # ring 1 starts at 1, ring 2 starts at 9, ...
        j = idx - ring_start               # local offset in [0, 8n-1]

        # Segment 1: (0,n) with 2 kinds
        if j < 2:
            kind = FourierDictionary._KINDS_KX0[j]
            return 0, n, kind

        j -= 2

        # Segment 2: (k,n) for k=1..n, each with 4 kinds
        # total length = 4n
        if j < 4 * n:
            k = 1 + (j // 4)                         # kx
            kind = FourierDictionary._KINDS_4[j % 4]
            return k, n, kind

        j -= 4 * n

        # Segment 3: (n,ky) for ky=n-1..1, each with 4 kinds
        # total length = 4*(n-1)
        if n > 1 and j < 4 * (n - 1):
            m = j // 4                               # 0..n-2
            ky = (n - 1) - m
            kind = FourierDictionary._KINDS_4[j % 4]
            return n, ky, kind

        j -= 4 * max(n - 1, 0)

        # Segment 4: (n,0) with 2 kinds
        if j < 2:
            kind = FourierDictionary._KINDS_KY0[j]
            return n, 0, kind

        raise IndexError(f"idx={idx} out of range for computed ring n={n}")

    def _get_atom(self, xy: torch.Tensor, idx: int):
        kx, ky, kind = self.idx_to_params(idx)
        return self._get_atom_by_args(xy, kx=kx, ky=ky, kind=kind)


class RadialFourierDictionary(FourierDictionary):
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return sample_from_disk(N)
    
    def _get_atom_by_args(
        self, 
        xy: torch.Tensor, 
        kx: int, 
        ky: int,
        kind: Literal["cc", "cs", "sc", "ss"] = "cc",
    ):
        r = xy[:, 0]**2 + xy[:, 1]**2
        theta = torch.atan2(xy[:, 1], xy[:, 0]) + math.pi
        theta = theta / (2 * math.pi)  # normalize to [-1,1]
        
        rtheta = torch.stack([r, theta], dim=1)
        return super()._get_atom_by_args(rtheta, kx, ky, kind)

