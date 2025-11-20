import torch
import math

import numpy as np
import torch


# sample a tensor (N, 2) which are the r and theta coordinates of a unit disk and the points are uniform on the disk, accounting for the Jacobian
def sample_uniform_disk(N):
    u = torch.rand(N, 2)
    r = torch.sqrt(u[:, 0])  # radius
    theta = 2 * math.pi * u[:, 1]  # angle
    return torch.stack([r, theta], dim=1)  # shape (N, 2)

def polar_to_cartesian(rtheta):
    r = rtheta[:, 0]
    theta = rtheta[:, 1]
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)  # shape (N, 2)

def random_bandpass_polar(
    xy, 
    min_k=0, 
    max_k=5, 
    max_n_radial=3, 
    decay_angle=0.0, 
    decay_radial=0.0, 
    seed=None,
):
    """
    Sample a random low-frequency function on the disk and evaluate it at specified points.
    
    Parameters
    ----------
    xy : torch.Tensor
        Tensor of shape (N, 2), where each row contains the polar coordinates [r, θ].
    min_k : int
        Minimum angular frequency.
    max_k : int
        Maximum angular frequency.
    max_n_radial : int
        Maximum radial frequency.
    decay_angle : float
        Controls how variance decays with angular frequency.
    decay_radial : float
        Controls how variance decays with radial frequency.
    seed : int or None
        Random seed for reproducibility.
        
    Returns
    -------
    torch.Tensor
        Function values evaluated at the given xy coordinates.
    """
    device = xy.device
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Extract radial (r) and angular (theta) coordinates from the cartesian input
    r = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    theta = torch.atan2(xy[:, 1], xy[:, 0])

    # Precompute the radial basis R_n(r) = cos(n * pi * r / r_max)
    r_max = r.max()
    radial_basis = torch.zeros((max_n_radial + 1, len(r))).to(device)
    for n in range(max_n_radial + 1):
        radial_basis[n, :] = torch.cos(n * np.pi * r / r_max)

    # Precompute cos(m*theta) and sin(m*theta) for each m
    cos_m = {}
    sin_m = {}
    for m in range(min_k, max_k + 1):
        cos_m[m] = torch.cos(m * theta)
        sin_m[m] = torch.sin(m * theta)

    # Initialize the function values as zeros
    f = torch.zeros_like(r)

    # Iterate over angular and radial frequencies
    for m in range(min_k, max_k + 1):
        for n in range(max_n_radial + 1):
            # Variance decay with frequency (optional)
            s_angle = (1.0 + abs(m)) ** decay_angle
            s_rad = (1.0 + n) ** decay_radial
            scale = 1.0 / (s_angle * s_rad)

            # Random Fourier coefficients
            a = np.random.normal(scale=scale)  # cosine coefficient
            if m == 0:
                # sin(0 theta) = 0, so only cosine part
                term = radial_basis[n, :] * a * cos_m[m]
            else:
                b = np.random.normal(scale=scale)  # sine coefficient
                term = radial_basis[n, :] * (a * cos_m[m] + b * sin_m[m])

            # Add the term to the function values
            f += term

    return f


def random_bandpass_field(xy, seed, f_lo, f_hi, n_waves=256):
    """
    Vectorized evaluation of a random band-pass 2D field at given coordinates.

    Parameters
    ----------
    xy : array-like, shape (N, 2)
        Coordinates in [0,1]^2 where the field is evaluated.
    seed : int
        Random seed that determines the random function.
    f_lo, f_hi : float
        Lower/upper bounds of the spatial frequency *band* (cycles per unit).
        Must satisfy 0 <= f_lo < f_hi.
    n_waves : int, optional
        Number of random plane waves to mix (controls texture/variance).

    Returns
    -------
    y : np.ndarray, shape (N,)
        Field values at each input coordinate.
    """
    device = xy.device

    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2)")
    if not (0.0 <= f_lo < f_hi):
        raise ValueError("Require 0 <= f_lo < f_hi")

    rng = torch.Generator().manual_seed(seed)

    # Random directions on the unit circle
    theta = torch.rand(size=(n_waves,), generator=rng) * 2.0 * torch.pi
    dirs = torch.column_stack((torch.cos(theta), torch.sin(theta)))  # (M,2)
    dirs = dirs.to(device)
    theta = theta.to(device)

    # Sample frequencies with density ∝ r so that wavevectors are ~uniform over the annulus area
    # Equivalent to sampling r^2 uniformly in [f_lo^2, f_hi^2].
    r2 = torch.rand(size=(n_waves,), generator=rng) * (f_hi*f_hi - f_lo*f_lo) + f_lo*f_lo
    r = torch.sqrt(r2).to(device)  # (M,)
    k = dirs * r[:, None]  # (M,2) wavevectors
    k = k.to(device)

    # Random phases and weights
    # phi = torch.rand(size=(n_waves,), generator=rng) * 2.0 * torch.pi            # (M,)
    weights = torch.randn(size=(n_waves,), generator=rng) / math.sqrt(n_waves)  # (M,)
    weights = weights.to(device)

    # Evaluate: sum_j w_j * cos(2π * <k_j, x> + φ_j)
    # phase = 2.0*torch.pi * (xy @ k.T) + phi  # (N,M)
    # y = (weights * torch.cos(phase)).sum(axis=1)  # (N,)

    y = (weights * torch.cos(xy @ k.T)).sum(axis=1)  # (N,)
    y = y.to(device)
    return y


def radial_basis(xy, n, m, indicator):
    """
    Compute the values of the radial-angular basis functions on the unit disk.

    Parameters
    ----------
    xy : torch.Tensor
        An N x 2 tensor where the first column is the x value and the second column is the y value
    n : int
        The radial index.
    m : int
        The angular index.
    indicator : int
        Determines which function to return:
        - 0: r^n * cos(m*theta)
        - 1: r^n * sin(m*theta)

    Returns
    -------
    torch.Tensor
        Values of the chosen basis function evaluated at the input points.
    """
    # Extract r and theta from xy tensor
    r = torch.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
    theta = torch.atan2(xy[:, 1], xy[:, 0])

    # Compute the radial part: r^n
    radial_part = r ** n

    # Compute the angular part
    if indicator == 0:
        # Cosine component: r^n * cos(m*theta)
        angular_part = torch.cos(m * theta)
    elif indicator == 1:
        # Sine component: r^n * sin(m*theta)
        angular_part = torch.sin(m * theta)
    else:
        raise ValueError("Indicator must be 0 (cosine) or 1 (sine).")

    # Combine radial and angular components
    
    return radial_part * angular_part * math.sqrt(2 * n + 2)


def tent_basis_2d(xy, row, col, total_rows, total_cols,
                  normalize="l2",  # 'l1' for ∫f=1, 'l2' for ‖f‖₂=1, or None
                  dtype=torch.float32):
    xy = torch.as_tensor(xy, dtype=dtype)

    w = 1.0 / float(total_cols)
    h = 1.0 / float(total_rows)
    xc = (col + 0.5) * w
    yc = (row + 0.5) * h

    u = torch.abs(xy[:, 0] - xc) / (0.5 * w)
    v = torch.abs(xy[:, 1] - yc) / (0.5 * h)
    tent = torch.clamp(1.0 - torch.maximum(u, v), min=0.0)

    if normalize is None:
        scale = 1.0
    elif normalize.lower() in ("l1", "integral"):
        scale = 3.0 / (w * h)                     # ∫ f = 1
    elif normalize.lower() in ("l2",):
        scale = math.sqrt(6.0 / (w * h))          # ‖f‖₂ = 1
    else:
        raise ValueError("normalize must be one of {'l1','l2',None}")

    return tent * scale


def fourier_basis_2d(xy, kx, ky, kind="cc", dtype=torch.float32):
    """
    L2-orthonormal real Fourier basis on [0,1]^2.
    xy:   (N, 2) points in [0,1]^2
    kx, ky: nonnegative integers (frequencies)
    kind: 'cc', 'cs', 'sc', or 'ss' for cos/cos, cos/sin, sin/cos, sin/sin.

    Returns: (N,) tensor with L2 norm = 1 over the unit square.
    """
    if kx < 0 or ky < 0:
        raise ValueError("kx, ky must be >= 0")
    if kind not in ("cc", "cs", "sc", "ss"):
        raise ValueError("kind must be one of {'cc','cs','sc','ss'}")

    xy = torch.as_tensor(xy, dtype=dtype)
    x, y = xy[:, 0], xy[:, 1]
    wx = 2.0 * math.pi * kx
    wy = 2.0 * math.pi * ky

    # 1D factors with correct L2 scaling on [0,1]
    def cos1d(k, w, t):
        return (1.0 if k == 0 else math.sqrt(2.0)) * torch.cos(w * t)

    def sin1d(k, w, t):
        # k==0 => identically zero function
        return (0.0 if k == 0 else math.sqrt(2.0)) * torch.sin(w * t)

    fx_c = cos1d(kx, wx, x); fx_s = sin1d(kx, wx, x)
    fy_c = cos1d(ky, wy, y); fy_s = sin1d(ky, wy, y)

    if kind == "cc":
        vals = fx_c * fy_c
    elif kind == "cs":
        vals = fx_c * fy_s
    elif kind == "sc":
        vals = fx_s * fy_c
    else:  # "ss"
        vals = fx_s * fy_s

    return vals  # shape (N,)

