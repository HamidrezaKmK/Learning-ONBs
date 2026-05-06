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
    (https://www.kaggle.com/datasets/qimaqi/implicit-zoo-cifar10-inrs).
    Reads directly from the zip — no extraction needed::

        cifar_inrs_dataset/{train,eval}/ckpts/<class_id>/<idx>_0000042.ckpt

    Each ``.ckpt`` is a dict ``{'image_idx': int, 'params': {seq.*}}``.

    CIFAR-10 architecture: 2 → 64 → 64 → 3,  SIREN ω₀ = 30.

Both classes share the same ``_eval_siren_batch`` kernel and expose the
``IrregularDataset`` API (``get_batch`` / ``reset_buffer``).
Coordinates are rescaled [0,1]² → [−1,1]² internally before evaluation.
"""

import os
import io
import glob
import zipfile
import torch
from torch import Tensor
from typing import Optional

from infidictionary.datasets.base import IrregularDataset
from infidictionary.domain_samplers import SquareSampler


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
# ImplicitZooMNISTDataset
# ---------------------------------------------------------------------------

class ImplicitZooMNISTDataset(IrregularDataset):
    """SIREN dataset from the Implicit-Zoo MNIST collection (directory format).

    Download
    --------
    https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0
    → download ``mnist-inrs.zip``, extract to ``data/mnist-inrs/``.

    Pass ``data_dir='data/mnist-inrs'`` and optionally a ``filter_pattern``
    to select only MNIST or only CIFAR-10 entries (see below).
    """

    def __init__(
        self,
        data_dir: str,
        domain_sample_size: int = 28,
        n_inrs: Optional[int] = None,
        filter_pattern: str = "mnist_png*",
        omega_0: float = 30.0,
        omega: float = 30.0,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            data_dir:           Root of the extracted ``mnist-inrs/`` directory.
            domain_sample_size: Samples *per dimension*; total coords = ``n²``.
                                28 → 784 coords, matching MNIST 28×28 resolution.
            n_inrs:             INRs to load (``None`` = all matching files).
            filter_pattern:     Glob pattern applied to sub-directory names.
                                ``'mnist_png*'``    → only MNIST (default).
                                ``'cifar10_png*'``  → only CIFAR-10.
                                ``'*'``             → all.
            omega_0:            First-layer angular frequency (default 30).
            omega:              Hidden-layer angular frequency (default 30).
            device:             Torch device.
        """
        self.domain_sample_size = domain_sample_size
        self.omega_0 = omega_0
        self.omega = omega
        self.device = torch.device(device)
        self._sampler = SquareSampler()

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Directory not found: {data_dir!r}\n"
                "Download mnist-inrs.zip from:\n"
                "  https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0\n"
                "and extract it."
            )

        # Collect model_final.pth files matching the filter
        pattern = os.path.join(data_dir, filter_pattern, "checkpoints", "model_final.pth")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(
                f"No files matched pattern {pattern!r}. "
                f"Check data_dir and filter_pattern."
            )

        # Shuffle before slicing so any n_inrs cut is class-balanced.
        # Lexicographic sort groups all class-0 files first, so without this
        # a small n_inrs would only load a single digit class.
        import random as _rng
        _rng.Random(0).shuffle(files)

        if n_inrs is not None:
            files = files[:n_inrs]

        self.weights, self.biases = self._load_files(files)
        self.n_inrs = self.weights[0].shape[0]

    def _load_files(self, files: list[str]) -> tuple[list[Tensor], list[Tensor]]:
        all_ws: list[list[Tensor]] = []
        all_bs: list[list[Tensor]] = []
        for path in files:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            ws, bs = _parse_seq_state_dict(sd)
            all_ws.append(ws)
            all_bs.append(bs)
        n_layers = len(all_ws[0])
        stacked_w = [torch.stack([all_ws[i][l] for i in range(len(files))]).to(self.device) for l in range(n_layers)]
        stacked_b = [torch.stack([all_bs[i][l] for i in range(len(files))]).to(self.device) for l in range(n_layers)]
        return stacked_w, stacked_b

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

        w_batch = [W[idx] for W in self.weights]
        b_batch = [b[idx] for b in self.biases]

        # SIRENs are trained with pixel-row order: x left→right, y top→bottom.
        # Our coords are in [0,1]² with y=0 at the bottom (mathematical convention),
        # so we flip y when mapping to the SIREN's native [-1,1]² range.
        siren_coords = torch.stack([
            2.0 * coords[:, 0] - 1.0,   # x: [0,1] → [-1,1]
            1.0 - 2.0 * coords[:, 1],   # y: [0,1] → [1,-1]  (flip)
        ], dim=-1)
        with torch.no_grad():
            values = _eval_siren_batch(siren_coords, w_batch, b_batch, self.omega_0, self.omega)

        return coords, values


# ---------------------------------------------------------------------------
# ImplicitZooCIFARDataset
# ---------------------------------------------------------------------------

class ImplicitZooCIFARDataset(IrregularDataset):
    """SIREN dataset from the Implicit-Zoo CIFAR-10 collection (zip format).

    Reads directly from the zip archive — no prior extraction needed.

    Download
    --------
    https://www.kaggle.com/datasets/qimaqi/implicit-zoo-cifar10-inrs
    → download ``archive.zip``, keep it as-is (pass its path to ``zip_path``).

    Each ``.ckpt`` entry inside the zip is a dict::

        {'image_idx': int, 'params': {'seq.0.weight': ..., ...}}

    CIFAR-10 architecture: 2 → 64 → 64 → 3,  ω₀ = 30.
    """

    def __init__(
        self,
        zip_path: str,
        split: str = "train",
        domain_sample_size: int = 32,
        n_inrs: Optional[int] = None,
        omega_0: float = 30.0,
        omega: float = 30.0,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            zip_path:           Path to ``archive.zip`` (Kaggle download).
            split:              ``'train'`` (50 000) or ``'eval'`` (10 000).
            domain_sample_size: Samples *per dimension*; total coords = ``n²``.
                                32 → 1024 coords, matching CIFAR-10 32×32.
            n_inrs:             INRs to load (``None`` = all in the chosen split).
            omega_0:            First-layer angular frequency (default 30).
            omega:              Hidden-layer angular frequency (default 30).
            device:             Torch device.
        """
        self.domain_sample_size = domain_sample_size
        self.omega_0 = omega_0
        self.omega = omega
        self.device = torch.device(device)
        self._sampler = SquareSampler()

        if not os.path.isfile(zip_path):
            raise FileNotFoundError(
                f"Zip archive not found: {zip_path!r}\n"
                "Download from https://www.kaggle.com/datasets/qimaqi/implicit-zoo-cifar10-inrs"
            )

        self.weights, self.biases = self._load_from_zip(zip_path, split, n_inrs)
        self.n_inrs = self.weights[0].shape[0]

    def _load_from_zip(
        self, zip_path: str, split: str, n_inrs: Optional[int]
    ) -> tuple[list[Tensor], list[Tensor]]:
        prefix = f"cifar_inrs_dataset/{split}/ckpts/"
        with zipfile.ZipFile(zip_path) as zf:
            ckpt_names = sorted(n for n in zf.namelist() if n.startswith(prefix) and n.endswith(".ckpt"))
            # Shuffle before slicing: zip entries sort by class directory, so any
            # n_inrs cut would otherwise contain only the first class (airplanes).
            import random as _rng
            _rng.Random(0).shuffle(ckpt_names)
            if n_inrs is not None:
                ckpt_names = ckpt_names[:n_inrs]

            all_ws: list[list[Tensor]] = []
            all_bs: list[list[Tensor]] = []
            for name in ckpt_names:
                with zf.open(name) as f:
                    obj = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)
                ws, bs = _parse_seq_state_dict(obj)
                all_ws.append(ws)
                all_bs.append(bs)

        n_layers = len(all_ws[0])
        n = len(ckpt_names)
        stacked_w = [torch.stack([all_ws[i][l] for i in range(n)]).to(self.device) for l in range(n_layers)]
        stacked_b = [torch.stack([all_bs[i][l] for i in range(n)]).to(self.device) for l in range(n_layers)]
        return stacked_w, stacked_b

    def reset_buffer(self) -> None:
        self._buffer = torch.randperm(self.n_inrs, device=self.device)
        self._buffer_ptr = 0

    def get_batch(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """Return ``batch_size`` SIRENs evaluated at fresh random coordinates.

        Returns:
            coords: (N, 2)    in [0, 1]².
            values: (B, N, 3) RGB values.
        """
        coords = self._sampler.sample(self.domain_sample_size).to(self.device)

        if hasattr(self, "_buffer"):
            if self._buffer_ptr + batch_size > self.n_inrs:
                self.reset_buffer()
            idx = self._buffer[self._buffer_ptr : self._buffer_ptr + batch_size]
            self._buffer_ptr += batch_size
        else:
            idx = torch.randint(self.n_inrs, (batch_size,), device=self.device)

        w_batch = [W[idx] for W in self.weights]
        b_batch = [b[idx] for b in self.biases]

        # Same flip as ImplicitZooMNISTDataset: y top→bottom in SIREN convention.
        siren_coords = torch.stack([
            2.0 * coords[:, 0] - 1.0,   # x: [0,1] → [-1,1]
            1.0 - 2.0 * coords[:, 1],   # y: [0,1] → [1,-1]  (flip)
        ], dim=-1)
        with torch.no_grad():
            values = _eval_siren_batch(siren_coords, w_batch, b_batch, self.omega_0, self.omega)

        return coords, values

