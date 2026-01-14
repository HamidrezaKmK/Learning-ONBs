
import torch
import math
from typing import Literal

from .base import InfiDictionary
from .utils import sample_from_disk

def cos1d(k: torch.Tensor, t: torch.Tensor):
    """
    k: (N_idx,) integer tensor
    t: (N_coords,) float tensor
    returns: (N_idx, N_coords)
    """
    k = k.to(t.dtype)
    kt = 2.0 * math.pi * k[:, None] * t[None, :]   # (N_idx, N_coords)

    out = math.sqrt(2.0) * torch.cos(kt)
    mask = (k > 0)[:, None]

    return torch.where(mask, out, torch.ones_like(out))


def sin1d(k: torch.Tensor, t: torch.Tensor):
    """
    k: (N_idx,) integer tensor
    t: (N_coords,) float tensor
    returns: (N_idx, N_coords)
    """
    k = k.to(t.dtype)
    kt = 2.0 * math.pi * k[:, None] * t[None, :]

    out = math.sqrt(2.0) * torch.sin(kt)
    mask = (k > 0)[:, None]

    return torch.where(mask, out, torch.zeros_like(out))

class FourierDictionary(InfiDictionary):

    def sample_from_domain(
        self,
        N: int,
    ):
        return torch.rand((N, 2))

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

    def idxs_to_params_vectorized(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (N,) int tensor (global indices)
        returns: (N,4) long tensor: [kx, ky, kind0, kind1]
        kind0 = 0 if first char is 'c' else 1
        kind1 = 0 if second char is 'c' else 1
        """
        if idx.ndim != 1:
            idx = idx.reshape(-1)
        if (idx < 0).any():
            raise ValueError("idx must be >= 0")

        device = idx.device
        idx = idx.to(torch.long)

        N = idx.numel()
        kx = torch.zeros(N, dtype=torch.long, device=device)
        ky = torch.zeros(N, dtype=torch.long, device=device)
        kind0 = torch.zeros(N, dtype=torch.long, device=device)
        kind1 = torch.zeros(N, dtype=torch.long, device=device)

        # n=0 special case: idx==0 -> (0,0,"cc") => bits (0,0)
        mask0 = (idx == 0)
        mask = ~mask0
        if mask.any():
            idx1 = idx[mask]

            # n = ceil((-1 + sqrt(1+idx))/2), clamp to >=1
            idxf = idx1.to(torch.float64)
            n = torch.ceil((-1.0 + torch.sqrt(1.0 + idxf)) / 2.0).to(torch.long)
            n = torch.clamp(n, min=1)

            ring_start = 1 + 4 * (n - 1) * n           # start index of ring n
            j = idx1 - ring_start                       # local offset in [0, 8n-1]

            # Segment 1: j in {0,1}  -> (0,n) kinds: ["cc","cs"]
            m1 = (j < 2)
            if m1.any():
                ky[mask.nonzero().squeeze(1)[m1]] = n[m1]
                # kind0 stays 0; kind1 is 0 for "cc", 1 for "cs"
                kind1[mask.nonzero().squeeze(1)[m1]] = j[m1]

            # Segment 2: next 4n -> (k,n), k=1..n, kinds: ["cc","cs","sc","ss"]
            j2 = j - 2
            m2 = (j2 >= 0) & (j2 < 4 * n)
            if m2.any():
                k = 1 + (j2[m2] // 4)
                r = (j2[m2] % 4)                        # 0..3
                out_idx = mask.nonzero().squeeze(1)[m2]
                kx[out_idx] = k
                ky[out_idx] = n[m2]
                kind0[out_idx] = (r >= 2).to(torch.long) # c,c,s,s
                kind1[out_idx] = (r & 1).to(torch.long)  # c,s,c,s

            # Segment 3: next 4(n-1) -> (n,ky), ky=n-1..1, same 4 kinds
            j3 = j2 - 4 * n
            m3 = (n > 1) & (j3 >= 0) & (j3 < 4 * (n - 1))
            if m3.any():
                m = (j3[m3] // 4)                       # 0..n-2
                ky3 = (n[m3] - 1) - m                   # n-1..1
                r = (j3[m3] % 4)
                out_idx = mask.nonzero().squeeze(1)[m3]
                kx[out_idx] = n[m3]
                ky[out_idx] = ky3
                kind0[out_idx] = (r >= 2).to(torch.long)
                kind1[out_idx] = (r & 1).to(torch.long)

            # Segment 4: last 2 -> (n,0) kinds: ["cc","sc"]
            # local index within seg4:
            j4 = j3 - 4 * (n - 1)
            m4 = (j4 >= 0) & (j4 < 2)
            if m4.any():
                out_idx = mask.nonzero().squeeze(1)[m4]
                kx[out_idx] = n[m4]
                # ky stays 0
                # kind0: 0 for "cc", 1 for "sc"; kind1 stays 0
                kind0[out_idx] = j4[m4]

            # sanity (optional): every nonzero idx must fall into exactly one segment
            # if you want strict checking, uncomment:
            # covered = m1 | m2 | m3 | m4
            # if not covered.all():
            #     bad = idx1[~covered]
            #     raise IndexError(f"Some idx out of range? e.g. {bad[:10].tolist()}")

        return torch.stack([kx, ky, kind0, kind1], dim=-1)


    def _get_atoms(self, xy: torch.Tensor, idx: torch.Tensor):
        kx_ky_kind = self.idxs_to_params_vectorized(idx)
        kx, ky, kind = kx_ky_kind[:,0], kx_ky_kind[:,1], kx_ky_kind[:,2]*2 + kx_ky_kind[:,3]
        kx = kx.to(xy.device)
        ky = ky.to(xy.device)
        kind = kind.to(xy.device)

        x, y = xy[:, 0], xy[:, 1]
        fx_c = cos1d(kx, x); fx_s = sin1d(kx, x)
        fy_c = cos1d(ky, y); fy_s = sin1d(ky, y)
        vals = torch.zeros((idx.shape[0], xy.shape[0]), device=xy.device, dtype=xy.dtype)

        if (kind == 0).any():
            vals[kind == 0] = fx_c[kind == 0] * fy_c[kind == 0]  # "cc"
        if (kind == 1).any():
            vals[kind == 1] = fx_c[kind == 1] * fy_s[kind == 1]  # "cs"
        if (kind == 2).any():
            vals[kind == 2] = fx_s[kind == 2] * fy_c[kind == 2]  # "sc"
        if (kind == 3).any():
            vals[kind == 3] = fx_s[kind == 3] * fy_s[kind == 3]  # "ss"

        return vals  # shape (N_idx, N_coords)


class RadialFourierDictionary(FourierDictionary):
    
    def sample_from_domain(
        self,
        N: int,
    ):
        return sample_from_disk(N)
    
    def _get_atoms(self, xy: torch.Tensor, idx: torch.Tensor):
        r = xy[:, 0]**2 + xy[:, 1]**2
        theta = torch.atan2(xy[:, 1], xy[:, 0]) + math.pi
        theta = theta / (2 * math.pi)  # normalize to [0, 1]
        
        rtheta = torch.stack([r, theta], dim=1)
        return super()._get_atoms(rtheta, idx)

