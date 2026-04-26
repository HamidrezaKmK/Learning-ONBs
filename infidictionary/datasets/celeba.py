import torch
from torchvision import transforms

from infidictionary.datasets.base import IrregularDataset


class CelebADataset(IrregularDataset):
    """
    Pre-loads `n_images` images from the CelebA-HQ dataset and exposes them
    as an IrregularDataset.

    `n_drop_draws` independent pixel-dropping patterns are pre-computed at
    construction time (seeded, reproducible).  Each pattern is drawn via
    independent Bernoulli sampling with keep-probability ``1 - drop_fraction``,
    so pattern lengths vary naturally across draws.

    The effective dataset size is ``n_images * n_drop_draws``: every
    (image, drop-pattern) pair is a distinct sample.

    get_batch selects one drop pattern uniformly at random, then draws
    `batch_size` images — all sharing that same irregular grid for the call.

    __call__(seed) is fully deterministic:
        image index = (seed // n_drop_draws) % n_images
        drop  index =  seed  % n_drop_draws
    """

    def __init__(
        self,
        resolution: int,
        n_images: int,
        drop_fraction: float = 0.0,
        n_drop_draws: int = 1,
        dataset_name: str = "mattymchen/celeba-hq",
        device: str | torch.device = "cpu",
        grayscale: bool = False,
    ):
        if not 0.0 <= drop_fraction < 1.0:
            raise ValueError("drop_fraction must be in [0, 1)")

        H = W = resolution
        N = H * W
        self.resolution = resolution
        self.n_images = n_images
        self.drop_fraction = drop_fraction
        self.n_drop_draws = n_drop_draws
        self.grayscale = grayscale
        self.channels = 1 if grayscale else 3
        self.device = torch.device(device)

        # Full pixel-grid coordinates in [0, 1]²  — (N, 2)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self._full_coords = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=-1
        ).to(self.device)  # (N, 2)

        # Pre-compute n_drop_draws fixed dropping patterns (seeded for reproducibility).
        # Each pattern is a Bernoulli draw: keep pixel i with probability (1 - drop_fraction).
        # Pattern lengths vary between draws.
        rng = torch.Generator()
        rng.manual_seed(42)
        if drop_fraction == 0.0:
            # No dropping: every pattern keeps all pixels
            pattern = torch.arange(N, device=self.device)
            self._drop_patterns = [pattern] * n_drop_draws
        else:
            self._drop_patterns = []
            for _ in range(n_drop_draws):
                mask = torch.rand(N, generator=rng) >= drop_fraction
                self._drop_patterns.append(mask.nonzero(as_tuple=False).squeeze(-1).to(self.device))

        self._drop_coords = [
            self._full_coords[pat] for pat in self._drop_patterns
        ]  # list of n_drop_draws tensors, each (n_keep_r, 2)  — lengths may differ

        # Pre-load images from the streaming HuggingFace dataset
        from datasets import load_dataset

        tfs = [
            transforms.Resize(
                (H, W), interpolation=transforms.InterpolationMode.BICUBIC
            ),
        ]
        if grayscale:
            # ITU-R BT.601 luminance: 0.299 R + 0.587 G + 0.114 B
            tfs.append(transforms.Grayscale(num_output_channels=1))
        tfs.append(transforms.ToTensor())  # -> (C, H, W) float32 in [0, 1]
        to_tensor = transforms.Compose(tfs)

        ds = load_dataset(dataset_name, split="train", streaming=True)
        it = iter(ds)
        signals = []
        for _ in range(n_images):
            img = next(it)["image"]
            t = to_tensor(img)              # (C, H, W) — C=1 grayscale, C=3 RGB
            t = t.permute(1, 2, 0)         # (H, W, C)
            t = t.flip(0)                  # flip vertically so image is upright
            signals.append(t.reshape(N, self.channels))

        self.images = torch.stack(signals, dim=0).to(self.device)  # (n_images, N, C)

    def reset_buffer(self) -> None:
        """Shuffle the image order and reset the cyclic pointer."""
        self._buffer = torch.randperm(self.n_images, device=self.device)
        self._buffer_ptr = 0

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `batch_size` images under one randomly selected drop pattern.

        All images in the batch share the same irregular grid (one pattern
        drawn uniformly from the pre-computed `n_drop_draws` patterns).

        Returns:
            coords : (N', 2)    — pixel coordinates for the selected pattern
            images : (B, N', C) — signals at those coordinates (C=1 grayscale, 3 RGB)
        """
        # Pick one drop pattern uniformly at random
        draw = int(torch.randint(self.n_drop_draws, (1,)).item())
        keep_idx = self._drop_patterns[draw]
        coords   = self._drop_coords[draw]

        # Image selection: cycle through buffer without overlap when available
        if hasattr(self, '_buffer'):
            if self._buffer_ptr + batch_size > self.n_images:
                self.reset_buffer()
            img_idx = self._buffer[self._buffer_ptr : self._buffer_ptr + batch_size]
            self._buffer_ptr += batch_size
        elif batch_size <= self.n_images:
            img_idx = torch.randperm(self.n_images, device=self.device)[:batch_size]
        else:
            img_idx = torch.randint(self.n_images, (batch_size,), device=self.device)

        images = self.images[img_idx][:, keep_idx, :]  # (B, N', C)
        return coords, images

    def __call__(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Deterministic lookup for a specific (image, drop-pattern) pair.

        seed is a flat index into the n_images × n_drop_draws grid:
            image index = (seed // n_drop_draws) % n_images
            drop  index =  seed  % n_drop_draws

        Returns:
            coords : (N', 2)  sparse pixel coordinates for the selected pattern
            vals   : (N', C)  values at those coords (C=1 grayscale, 3 RGB)
        """
        img_idx  = (seed // self.n_drop_draws) % self.n_images
        draw_idx = seed % self.n_drop_draws
        keep_idx = self._drop_patterns[draw_idx]
        return self._drop_coords[draw_idx], self.images[img_idx][keep_idx]
