import torch


def pairwise_inner_product(
    f1: torch.Tensor,                       # (A, N, C)
    f2: torch.Tensor,                       # (B, N, C)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (A, B)
    """Weighted L² inner product."""
    if logabsdet is None:
        logabsdet = torch.zeros(f1.shape[1], device=f1.device, dtype=f1.dtype)
    w = torch.exp(logabsdet).to(f1.dtype)
    ret = torch.einsum("anc, n, bnc -> ab", f1, w, f2)
    return ret / f1.shape[1]


def norm2(
    f: torch.Tensor,                        # (B, N, C)
    logabsdet: torch.Tensor | None = None,  # (N,)
) -> torch.Tensor:                          # (B,) — non-negative
    """Squared weighted L² norm."""
    if logabsdet is None:
        logabsdet = torch.zeros(f.shape[1], device=f.device, dtype=f.dtype)
    w = torch.exp(logabsdet).to(f.dtype)
    ret = torch.einsum("bnc, n, bnc -> b", f, w, f)
    return ret / f.shape[1]


def haar_orthogonal(n: int, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Haar-random ``(n, n)`` orthogonal matrix: QR of a Gaussian, sign-corrected."""
    q, r = torch.linalg.qr(torch.randn(n, n, dtype=dtype))
    signs = torch.sign(torch.diagonal(r))
    return q * torch.where(signs == 0, torch.ones_like(signs), signs)[None, :]
