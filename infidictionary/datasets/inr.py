"""Datasets of pre-trained SIRENs from the Implicit-Zoo collection.

Two concrete classes cover the two formats actually shipped:

``ImplicitZooMNISTDataset``
    Source: ``mnist-inrs.zip`` from the DWSNets / Implicit-Zoo Dropbox.
    Each SIREN lives in its own sub-directory::

        mnist-inrs/
          mnist_png_training_<class>_<idx>/checkpoints/model_final.pth
          cifar10_png_train_<class>_<idx>/checkpoints/model_final.pth
          ...

    State-dict format (``OrderedDict``):
        ``seq.0.weight`` (out, in), ``seq.0.bias`` (out), ...

    MNIST architecture: 2 → 32 → 32 → 1,  SIREN ω₀ = 30.

``ImplicitZooCIFARDataset``
    Source: ``archive.zip`` from Kaggle
    (https://www.kaggle.com/datasets/alexanderqi/cifar10-inrs-dataset).
    The archive must be extracted before use::

        cifar_inrs_dataset/{train,eval}/ckpts/<class_id>/<idx>_0000042.ckpt

    Each ``.ckpt`` file is a dict ``{'image_idx': int, 'params': {seq.*}}``.

    CIFAR-10 architecture: 2 → 64 → 64 → 3,  SIREN ω₀ = 30.

Both classes share the same ``_eval_siren_batch`` kernel and expose the
``IrregularDataset`` API (``get_batch`` / ``reset_buffer``).
Coordinates are rescaled [0,1]² → [−1,1]² internally before evaluation.
"""

import glob
import os
import random

import torch
from torch import Tensor
from typing import Optional

from infidictionary.datasets.base import IrregularDataset
from infidictionary.domain_samplers import SquareSampler


def _resolve_device(device: str | torch.device) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# Shared SIREN batch evaluator
# ---------------------------------------------------------------------------

def _eval_siren_batch(
    coords: Tensor,
    weights: list[Tensor],
    biases: list[Tensor],
    omega_0: float = 30.0,
    omega: float = 30.0,
) -> Tensor:
    """Vectorized SIREN forward pass over a batch of B networks.

    Args:
        coords:  (N, d)  query coordinates already in the SIREN's native range.
        weights: L tensors, each (B, out_i, in_i).
        biases:  L tensors, each (B, out_i).
        omega_0: angular frequency for the first (input) layer.
        omega:   angular frequency for hidden layers; last layer is always linear.

    Returns:
        (B, N, out_L)
    """
    B = weights[0].shape[0]
    x = coords.unsqueeze(0).expand(B, -1, -1)   # (B, N, in_0)

    n = len(weights)
    for i, (W, b) in enumerate(zip(weights, biases)):
        x = torch.bmm(x, W.transpose(-1, -2)) + b.unsqueeze(1)
        if i == 0:
            x = torch.sin(omega_0 * x)
        elif i < n - 1:
            x = torch.sin(omega * x)
        # last layer: linear, no activation

    return x  # (B, N, out_L)


# ---------------------------------------------------------------------------
# Shared state-dict parser
# ---------------------------------------------------------------------------

def _parse_seq_state_dict(sd: dict) -> tuple[list[Tensor], list[Tensor]]:
    """Extract (weights, biases) from a ``seq.N.weight`` / ``seq.N.bias`` state dict.

    Handles two wrapping conventions:
    - Direct ``OrderedDict`` with ``seq.*`` keys  (MNIST format).
    - Nested ``{'params': {seq.*}}``               (CIFAR .ckpt format).
    """
    # Unwrap nested params dict
    if "params" in sd and isinstance(sd["params"], dict):
        sd = sd["params"]

    indices = sorted(
        set(int(k.split(".")[1]) for k in sd if k.startswith("seq.") and k.endswith(".weight"))
    )
    if not indices:
        raise ValueError(f"No seq.N.weight keys found. Keys: {list(sd.keys())[:8]}")

    weights = [sd[f"seq.{i}.weight"] for i in indices]
    biases  = [sd[f"seq.{i}.bias"]   for i in indices]
    return weights, biases


# ---------------------------------------------------------------------------
# Shared base: a stack of SIRENs evaluated at fresh random coordinates
# ---------------------------------------------------------------------------

class _SirenDataset(IrregularDataset):
    """Loads SIREN checkpoints into stacked ``weights`` / ``biases`` and serves batches.

    Subclasses only locate the checkpoint files. The files are shuffled with a
    fixed seed before an optional ``n_inrs`` cut, because they sort by class and
    a plain prefix would load only the first classes.
    """

    def __init__(
        self,
        files: list[str],
        domain_sample_size: int,
        n_inrs: Optional[int],
        omega_0: float,
        omega: float,
        device: str | torch.device,
    ):
        self.domain_sample_size = domain_sample_size
        self.omega_0 = omega_0
        self.omega = omega
        self.device = _resolve_device(device)
        self._sampler = SquareSampler()

        files = list(files)
        random.Random(0).shuffle(files)
        if n_inrs is not None:
            files = files[:n_inrs]
        params = [_parse_seq_state_dict(torch.load(f, map_location="cpu", weights_only=False)) for f in files]
        n_layers = len(params[0][0])
        self.weights = [torch.stack([w[l] for w, _ in params]).to(self.device) for l in range(n_layers)]
        self.biases = [torch.stack([b[l] for _, b in params]).to(self.device) for l in range(n_layers)]
        self.n_inrs = self.weights[0].shape[0]

    def reset_buffer(self) -> None:
        self._buffer = torch.randperm(self.n_inrs, device=self.device)
        self._buffer_ptr = 0

    def get_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return ``batch_size`` SIRENs evaluated at fresh random coordinates.

        Returns:
            coords: (N, 2)    in [0, 1]².
            values: (B, N, C) network outputs (C=1 for MNIST, C=3 for CIFAR).
        """
        coords = self._sampler.sample(self.domain_sample_size).to(self.device)

        if hasattr(self, "_buffer"):
            if self._buffer_ptr + batch_size > self.n_inrs:
                self.reset_buffer()
            idx = self._buffer[self._buffer_ptr : self._buffer_ptr + batch_size]
            self._buffer_ptr += batch_size
        else:
            idx = torch.randint(self.n_inrs, (batch_size,), device=self.device)

        # SIRENs are trained with pixel-row order: x left→right, y top→bottom.
        # Our coords are in [0,1]² with y=0 at the bottom (mathematical convention),
        # so we flip y when mapping to the SIREN's native [-1,1]² range.
        siren_coords = torch.stack([
            2.0 * coords[:, 0] - 1.0,   # x: [0,1] → [-1,1]
            1.0 - 2.0 * coords[:, 1],   # y: [0,1] → [1,-1]  (flip)
        ], dim=-1)
        with torch.no_grad():
            values = _eval_siren_batch(
                siren_coords,
                [W[idx] for W in self.weights],
                [b[idx] for b in self.biases],
                self.omega_0,
                self.omega,
            )
        return coords, values


# ---------------------------------------------------------------------------
# ImplicitZooMNISTDataset
# ---------------------------------------------------------------------------

class ImplicitZooMNISTDataset(_SirenDataset):
    """SIREN dataset from the Implicit-Zoo MNIST collection (directory format).

    Download ``mnist-inrs.zip`` (inside the DWSNets Dropbox bundle,
    https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0)
    and extract it to ``data/mnist-inrs/``.

    Args:
        data_dir:           Root of the extracted ``mnist-inrs/`` directory.
        domain_sample_size: Samples *per dimension*; total coords = ``n²``.
        n_inrs:             INRs to load (``None`` = all matching files).
        filter_pattern:     Glob over sub-directory names: ``'mnist_png*'`` (MNIST,
                            default), ``'cifar10_png*'`` (CIFAR-10) or ``'*'``.
        omega_0, omega:     First- and hidden-layer SIREN frequencies.
        device:             Torch device; ``'auto'`` picks CUDA when available.
    """

    def __init__(
        self,
        data_dir: str,
        domain_sample_size: int = 28,
        n_inrs: Optional[int] = None,
        filter_pattern: str = "mnist_png*",
        omega_0: float = 30.0,
        omega: float = 30.0,
        device: str | torch.device = "auto",
    ):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Directory not found: {data_dir!r}. Download mnist-inrs.zip from "
                "https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0 "
                "and extract it."
            )
        pattern = os.path.join(data_dir, filter_pattern, "checkpoints", "model_final.pth")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched {pattern!r}. Check data_dir and filter_pattern.")
        super().__init__(files, domain_sample_size, n_inrs, omega_0, omega, device)


# ---------------------------------------------------------------------------
# ImplicitZooCIFARDataset
# ---------------------------------------------------------------------------

class ImplicitZooCIFARDataset(_SirenDataset):
    """SIREN dataset from the Implicit-Zoo CIFAR-10 collection.

    Download ``archive.zip`` from
    https://www.kaggle.com/datasets/alexanderqi/cifar10-inrs-dataset and
    extract it into ``data/`` so that ``data/cifar_inrs_dataset/`` exists.

    Args:
        extracted_dir:      Root of the extracted ``cifar_inrs_dataset/`` tree.
        split:              ``'train'`` (50 000) or ``'eval'`` (10 000).
        domain_sample_size: Samples *per dimension*; total coords = ``n²``.
        n_inrs:             INRs to load (``None`` = all in the chosen split).
        omega_0, omega:     First- and hidden-layer SIREN frequencies.
        device:             Torch device; ``'auto'`` picks CUDA when available.
    """

    def __init__(
        self,
        extracted_dir: str,
        split: str = "train",
        domain_sample_size: int = 32,
        n_inrs: Optional[int] = None,
        omega_0: float = 30.0,
        omega: float = 30.0,
        device: str | torch.device = "auto",
    ):
        files = sorted(glob.glob(os.path.join(extracted_dir, split, "ckpts", "*", "*.ckpt")))
        if not files:
            raise FileNotFoundError(
                "No CIFAR INR checkpoints found under "
                f"{os.path.join(extracted_dir, split, 'ckpts')!r}. Download and extract the archive from "
                "https://www.kaggle.com/datasets/alexanderqi/cifar10-inrs-dataset"
            )
        super().__init__(files, domain_sample_size, n_inrs, omega_0, omega, device)
