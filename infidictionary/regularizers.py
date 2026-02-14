import torch
from abc import ABC, abstractmethod
from PIL import Image
import torch.nn.functional as F
import numpy as np

class Regularizer(ABC):
    @abstractmethod
    def compute_energy(self, coords: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        pass

class EntropyRegularizer(Regularizer):
    def __init__(self, sigma: float = 0.01):
        self.sigma = sigma

    def compute_energy(self, coords: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        # F is (A, N)
        # We want to estimate p(x) for each row
        
        # 1. Calculate pairwise differences between all values in a single signal
        # (A, N, 1) - (A, 1, N) -> (A, N, N)
        val_diffs = F.unsqueeze(-1) - F.unsqueeze(-2)
        
        # 2. Gaussian Kernel to estimate density
        # This acts like a "soft histogram"
        kde = torch.exp(-val_diffs.pow(2) / (2 * self.sigma**2))
        
        # 3. Density p(x) is the mean of the kernel values
        p_x = kde.mean(dim=-1) + 1e-8 # (A, N)
        
        # 4. Entropy H(x) = - sum(p(x) * log(p(x)))
        # Since we are sampling x from the signal itself, we approximate the integral via mean
        entropy = -torch.mean(torch.log(p_x), dim=-1)
        
        return entropy

class TVRegularizer(Regularizer):
    def __init__(
        self, 
        sigma: float = 0.1,
        neighbourhood_r: float = 0.1

    ):
        self.sigma = sigma
        self.neighbourhood_r = neighbourhood_r


    def compute_energy(self, coords: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        
        distances = torch.cdist(coords, coords)
        weights = torch.where(
            distances < self.neighbourhood_r, 
            torch.exp(-distances.pow(2) / (2 * self.sigma ** 2)), 
            torch.zeros_like(distances)
        )
        weights.fill_diagonal_(0)

        # F is (A, N)
        # We want to compute the total variation of each signal
        
        # 1. Calculate pairwise differences between all values in a single signal
        # (A, N, 1) - (A, 1, N) -> (A, N, N)
        val_diffs = F.unsqueeze(-1) - F.unsqueeze(-2)
        

        energy = torch.mean(weights * torch.abs(val_diffs), dim=(1, 2))
        
        return energy

class GraphLaplacianRegularizer(Regularizer):
    def __init__(
        self,
        sigma: float = 0.1,
        neighbourhood_r: float = 0.1
    ):
        self.sigma = sigma
        self.neighbourhood_r = neighbourhood_r

    def compute_energy(self, coords: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(coords, coords)
        weights = torch.where(
            distances < self.neighbourhood_r, 
            torch.exp(-distances.pow(2) / (2 * self.sigma ** 2)), 
            torch.zeros_like(distances)
        )
        weights.fill_diagonal_(0)

        laplacian = torch.diag(weights.sum(dim=1)) -  weights
        energy = torch.einsum("ij,jk,ik->i", F, laplacian, F) / laplacian.shape[0] # shape (A, )

        return energy




def load_bw_tensor_from_path(
            path: str = "../assets/Joseph_Fourier.jpg",
            resize_to: int | None = 512,
            device: torch.device | str = "cpu",
            dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            """
            Returns: (1, 1, H, W) grayscale in [0,1]
            """
            img = Image.open(path).convert("L")  # grayscale
            if resize_to is not None:
                W, H = img.size
                scale = resize_to / max(W, H)
                newW, newH = int(round(W * scale)), int(round(H * scale))
                img = img.resize((newW, newH), Image.BICUBIC)

            arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W)
            t = torch.from_numpy(arr).to(device=device, dtype=dtype)
            # standardize pixels to have mean 0 and std 1, to make the regularizer more stable to train
            t = (t - t.mean()) / (t.std() + 1e-8)
            return t.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

class FouriererRegularizer(Regularizer):

    def __init__(
        self,
        asset_path: str = "./assets/Joseph_Fourier.jpg",
        resize_to: int = 512,
    ):
        self.img = load_bw_tensor_from_path(asset_path, resize_to=resize_to, device="cpu")  # (1,1,H,W)
       

    def sample_image_irregular(
        self,
        img_1x1xHxW: torch.Tensor,
        xy: torch.Tensor,
        coords: str = "pixel",          # "pixel" or "unit"
        mode: str = "bilinear",
        padding_mode: str = "border",
        align_corners: bool = True,
    ) -> torch.Tensor:
        """
        xy: (N,2)
        - coords="pixel": x in [0,W-1], y in [0,H-1], origin TOP-LEFT
        - coords="unit":  x,y in [0,1], origin TOP-LEFT
        Returns: (N,)
        """
        _, _, H, W = img_1x1xHxW.shape
        xy = xy.to(device=img_1x1xHxW.device, dtype=img_1x1xHxW.dtype)

        if coords == "unit":
            x = xy[:, 0] * (W - 1)
            y =  (1 - xy[:, 1]) * (H - 1)
        elif coords == "pixel":
            x, y = xy[:, 0], (H - 1) - xy[:, 1]
        else:
            raise ValueError("coords must be 'pixel' or 'unit'")

        if align_corners:
            gx = 2.0 * (x / (W - 1)) - 1.0
            gy = 2.0 * (y / (H - 1)) - 1.0
        else:
            gx = (2.0 * x + 1.0) / W - 1.0
            gy = (2.0 * y + 1.0) / H - 1.0

        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)  # (1,N,1,2)

        samples = F.grid_sample(
            img_1x1xHxW, grid,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (1,1,N,1)

        return samples.view(-1)  # (N,)

    def compute_energy(self, coords: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
        # F is (A, N)
        # We want to compute the fidelity of each signal to the image values at the corresponding coordinates
        
        # 1. Sample the image at the given coordinates
        sampled_vals = self.sample_image_irregular(self.img.to(coords.device), coords, coords="unit")  # (N,)

        # 2. Compute fidelity as mean squared error between each signal and the sampled values
        fidelity = torch.mean((F - sampled_vals.unsqueeze(0)) ** 2, dim=1)  # shape (A, )

        return fidelity

    def get_joseph(self, coords: torch.Tensor) -> torch.Tensor:
        return self.sample_image_irregular(self.img.to(coords.device), coords, coords="unit")  # (N,)
