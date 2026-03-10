import torch
from torchvision import transforms

from infidictionary.datasets.base import IrregularDataset


class CelebADataset(IrregularDataset):
    """
    Pre-loads `n_images` images from the CelebA-HQ dataset and exposes them
    as an IrregularDataset.

    On each call to get_batch a random `drop_fraction` of the pixels are
    discarded, so the returned coordinates are an irregular subset of the
    full pixel grid.

    get_batch returns:
        coords : (N', 2)    sparse pixel-grid coordinates in [0, 1]²
        images : (B, N', 3) batch of RGB signals at those coordinates
    where N' = round(H * W * (1 - drop_fraction)).
    """

    def __init__(
        self,
        resolution: int,
        n_images: int,
        drop_fraction: float = 0.0,
        dataset_name: str = "mattymchen/celeba-hq",
        device: str | torch.device = "cpu",
    ):
        if not 0.0 <= drop_fraction < 1.0:
            raise ValueError("drop_fraction must be in [0, 1)")

        H = W = resolution
        N = H * W
        self.resolution = resolution
        self.n_images = n_images
        self.drop_fraction = drop_fraction
        self.n_keep = round(N * (1.0 - drop_fraction))
        self.device = torch.device(device)

        # Full pixel-grid coordinates in [0, 1]²  — (N, 2)
        xs = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W)
        ys = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self._full_coords = torch.stack(
            [grid_x.flatten(), grid_y.flatten()], dim=-1
        ).to(self.device)  # (N, 2)

        # Pre-load images from the streaming HuggingFace dataset
        from datasets import load_dataset

        to_tensor = transforms.Compose([
            transforms.Resize(
                (H, W), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),  # -> (3, H, W) float32 in [0, 1]
        ])

        ds = load_dataset(dataset_name, split="train", streaming=True)
        it = iter(ds)
        signals = []
        for _ in range(n_images):
            img = next(it)["image"]
            t = to_tensor(img)       # (3, H, W)
            t = t.permute(1, 2, 0)  # (H, W, 3)
            signals.append(t.reshape(N, 3))

        self.images = torch.stack(signals, dim=0).to(self.device)  # (n_images, N, 3)

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `batch_size` images and a random irregular subset of pixels.

        Returns:
            coords : (N', 2)    — randomly subsampled pixel coordinates
            images : (B, N', 3) — RGB signals at those coordinates
        where N' = round(H*W * (1 - drop_fraction)).
        """
        N_full = self._full_coords.shape[0]

        # Random pixel subset — new draw on every call, giving an irregular grid
        keep_idx = torch.randperm(N_full, device=self.device)[:self.n_keep]
        coords = self._full_coords[keep_idx]          # (N', 2)

        # Random image subset
        if batch_size <= self.n_images:
            img_idx = torch.randperm(self.n_images, device=self.device)[:batch_size]
        else:
            img_idx = torch.randint(self.n_images, (batch_size,), device=self.device)

        images = self.images[img_idx][:, keep_idx, :]  # (B, N', 3)

        return coords, images
