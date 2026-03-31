import torch


def pairwise_inner_product(
    f1: torch.Tensor, # (A, N, C) or (N, C)
    f2: torch.Tensor, # (B, N, C) or (N, C)
    logabsdet: torch.Tensor | None = None, # (N, )
) -> torch.Tensor:
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    first_unsqueezed = False
    second_unsqueezed = False
    if f1.dim() == 2:
        f1 = f1.unsqueeze(0)  # (1, N, C)
        first_unsqueezed = True
    if f2.dim() == 2:
        f2 = f2.unsqueeze(0)  # (1, N, C)
        second_unsqueezed = True
    ret = torch.einsum("anc, bnc->ab", f1 * torch.exp(logabsdet)[None, :, None], f2) # (A, B)
    ret = ret / f1.shape[1] # (A, B) or (A,) or (B,) or scalar
    if first_unsqueezed:
        ret = ret.squeeze(0)
    if second_unsqueezed:
        ret = ret.squeeze(-1)
    return ret

def norm2(
    f: torch.Tensor, # (B, N, C) or (N, C)
    logabsdet: torch.Tensor | None = None, # (N, )
) -> torch.Tensor:
    if f.dim() == 2:
        f = f.unsqueeze(0)  # (1, N, C)
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f.shape[1], device=f.device, dtype=f.dtype)
    ret = torch.einsum("bnc, bnc->b", f * torch.exp(logabsdet)[None, :, None], f) # (B, )
    return ret.squeeze() / f.shape[1]  # (B,) or scalar

def parallel_inner_product(
    f1: torch.Tensor, # (B, N, C)
    f2: torch.Tensor, # (B, N, C)
    logabsdet: torch.Tensor | None = None, # (B, N, ) or (N, ) or None
) -> torch.Tensor:
    logabsdet = logabsdet if logabsdet is not None else torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    if logabsdet.dim() == 1:
        logabsdet = logabsdet.unsqueeze(0).repeat(f1.shape[0], 1)  # (B, N)
    ret = torch.einsum("bnc, bnc->b", f1 * torch.exp(logabsdet)[:, :, None], f2) # (B, )
    return ret.squeeze() / f1.shape[1]  # (B,) or scalar
