
from .base import PushforwardRegularizer
import torch
import torch.nn.functional as F
import math

class ClassActivationRegularizer(PushforwardRegularizer):
    """Maximises a target CIFAR-10 class logit for each pushed-forward atom.

    Atoms are evaluated on a stratified ``image_size × image_size`` pixel grid
    built internally.  The resulting (A, 3, image_size, image_size) images are
    passed directly to a frozen CIFAR-10 classifier after optional Gaussian noise
    augmentation.

    The internal ``SquareSampler(stratified=True, add_noise=add_noise)`` produces
    coords in ``indexing='ij'`` order, so ``pushed.reshape(A, n, n, 3)`` is a valid
    spatial image with no KNN interpolation required.

    CIFAR-10 indices: 0 airplane 1 automobile 2 bird 3 cat 4 deer
                      5 dog 6 frog 7 horse 8 ship 9 truck

    Args:
        image_size:     Grid resolution; total quadrature points = image_size².
                        Must match the classifier's expected input size (32).
        add_noise:      If True, adds per-epoch jitter to the stratified grid.
        target_class:   CIFAR-10 class to maximise (default 3 = cat).
        hub_repo:       torch.hub source for the CIFAR-10 model.
        hub_model:      Model name within that repo.
        noise_std:      Std of Gaussian noise added per trial for robustness.
        n_noise_trials: Number of noised copies per atom per step.
    """

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    _MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    _STD  = torch.tensor([0.2023, 0.1994, 0.2010])

    def __init__(
        self,
        image_size: int = 32,
        add_noise: bool = False,
        target_class: int = 3,
        hub_repo: str = "chenyaofo/pytorch-cifar-models",
        hub_model: str = "cifar10_resnet20",
        noise_std: float = 0.05,
        n_noise_trials: int = 8,
    ):
        from infidictionary.domain_samplers import SquareSampler
        super().__init__(SquareSampler(stratified=True, add_noise=add_noise), image_size)
        assert 0 <= target_class <= 9
        self.image_size     = image_size
        self.target_class   = target_class
        self.noise_std      = noise_std
        self.n_noise_trials = n_noise_trials
        self.model          = torch.hub.load(hub_repo, hub_model, pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"ClassActivationRegularizer: target='{self.CIFAR10_CLASSES[target_class]}' "
              f"(class {target_class}), model={hub_model}, grid={image_size}x{image_size}, "
              f"noise_std={noise_std}, n_noise_trials={n_noise_trials}")

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        A, T   = pushed.shape[0], self.n_noise_trials
        n      = self.image_size
        device = pushed.device

        # Direct reshape: pushed (A, n², 3) → (A, 3, n, n)
        # Valid because SquareSampler uses indexing='ij': coords[i*n+j] = (x_i, y_j)
        grid = pushed.reshape(A, n, n, 3).permute(0, 3, 1, 2)   # (A, 3, n, n)

        # Clamp to [-1, 1] then rescale to [0, 1] for the classifier.
        grid_clipped = (grid.clamp(-1, 1) + 1) / 2              # (A, 3, n, n)

        # Tile T copies per atom and add independent Gaussian noise to each
        grid_clipped = grid_clipped.repeat_interleave(T, dim=0)  # (A*T, 3, n, n)
        grid_clipped = grid_clipped + torch.randn_like(grid_clipped) * self.noise_std

        mean  = self._MEAN.to(device)[None, :, None, None]
        std   = self._STD.to(device)[None, :, None, None]
        batch = (grid_clipped - mean) / std

        self.model.to(device)
        range_penalty = (F.relu(-1 - grid) + F.relu(grid - 1)).mean(dim=(1, 2, 3))  # (A,)

        logits = self.model(batch)                               # (A*T, 10)
        cls_energy = -logits[:, self.target_class].reshape(A, T).mean(dim=1)  # (A,)
        return cls_energy + range_penalty


class CLIPRegularizer(PushforwardRegularizer):
    """Maximises CLIP cosine similarity with a per-atom text embedding.

    A single prompt per color channel is encoded into a base embedding.  Each
    atom's target is that base embedding perturbed by a small noise vector drawn
    deterministically from a hash of its multi-index, then re-normalised.  This
    encourages diversity: different atoms are pulled towards distinct
    neighbourhoods of the CLIP text sphere while all remaining close to the base.

    The last index component (channel) selects the color embedding:
        0 → "Red",  1 → "Green",  2 → "Blue"

    Args:
        image_size:            Grid resolution; total quadrature points = image_size².
        add_noise:             If True, adds per-epoch jitter to the stratified grid.
        subject:               Subject noun used in the prompt (default "Cat").
        model_name:            OpenCLIP model architecture (default "ViT-B-32").
        pretrained:            OpenCLIP pretrained weights tag (default "openai").
        n_augmentation_trials: Independently augmented views per atom per step.
        bg_alpha:              Foreground opacity when compositing over random background.
        embedding_noise_std:   Std of per-atom noise added in CLIP embedding space
                               before re-normalisation.  0 disables diversity noise.
    """

    _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
    _CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    _COLORS    = ["red", "green", "blue"]
    _BANK_SIZE = 100_003   # prime — reduces hash collisions

    def __init__(
        self,
        image_size: int = 64,
        add_noise: bool = False,
        subject: str = "Cat",
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        n_augmentation_trials: int = 8,
        bg_alpha: float = 0.8,
        embedding_noise_std: float = 0.0,
        range_penalty_weight: float = 1.0,
    ):
        import open_clip
        from torchvision.transforms import v2 as T
        from infidictionary.domain_samplers import SquareSampler

        super().__init__(SquareSampler(stratified=True, add_noise=add_noise), image_size)
        self.image_size            = image_size
        self.n_augmentation_trials = n_augmentation_trials
        self.bg_alpha              = bg_alpha
        self.embedding_noise_std   = embedding_noise_std
        self.range_penalty_weight  = range_penalty_weight

        self._model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._clip_size: int = self._model.visual.image_size
        if isinstance(self._clip_size, (list, tuple)):
            self._clip_size = self._clip_size[0]

        # Precompute one text embedding per color and store on CPU.
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = [f"A {subject} colored in {color}, front view" for color in self._COLORS]
        with torch.no_grad():
            feats = F.normalize(self._model.encode_text(tokenizer(prompts)), dim=-1)  # (3, D)
        self._color_features = feats.cpu()   # (3, D)

        # Noise bank: large table of unit-normal vectors.  Each atom indexes into
        # it via a polynomial hash of its multi-index — deterministic and vectorised.
        D = self._color_features.shape[-1]
        bank_gen = torch.Generator()
        bank_gen.manual_seed(0)
        self._noise_bank = torch.randn(self._BANK_SIZE, D, generator=bank_gen)  # (B, D)

        self._augment = T.Compose([
            T.RandomResizedCrop(self._clip_size, scale=(0.25, 1.0), ratio=(0.5, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

        print(f"CLIPRegularizer: subject='{subject}', model={model_name}/{pretrained}, "
              f"clip_size={self._clip_size}, grid={image_size}x{image_size}, "
              f"n_augmentation_trials={n_augmentation_trials}, "
              f"embedding_noise_std={embedding_noise_std}")
        for color, prompt in zip(self._COLORS, prompts):
            print(f"  [{color}] {prompt}")

    def _random_bg(self, g: torch.Tensor) -> torch.Tensor:
        """Generate a random structured background (same shape as g).

        Randomly selects between uniform noise and a Fourier sinusoidal texture.
        Structured backgrounds force CLIP to respond to the spatial structure of
        the foreground rather than background colour alone (Dream Fields finding).
        """
        A, C, H, W = g.shape
        device = g.device
        if torch.randint(0, 2, (1,)).item() == 0:
            return torch.rand_like(g)
        # Fourier sinusoidal background
        freq = torch.randint(2, 6, (A, C, 1, 1), device=device).float()
        ph_x = torch.rand(A, C, 1, 1, device=device) * 2 * math.pi
        ph_y = torch.rand(A, C, 1, 1, device=device) * 2 * math.pi
        xs = torch.linspace(0, 2 * math.pi, W, device=device)[None, None, None, :]
        ys = torch.linspace(0, 2 * math.pi, H, device=device)[None, None, :, None]
        return (0.5 + 0.5 * torch.sin(freq * xs + ph_x) * torch.sin(freq * ys + ph_y)).clamp(0, 1)

    def _clip_normalise(self, batch: torch.Tensor) -> torch.Tensor:
        device = batch.device
        mean = self._CLIP_MEAN.to(device)[None, :, None, None]
        std  = self._CLIP_STD.to(device)[None,  :, None, None]
        return (batch - mean) / std

    def _atom_text_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Return (A, D) per-atom text features.

        Base embedding is selected by channel (last index component % 3).
        A deterministic per-atom noise vector — indexed via a polynomial hash
        of the full multi-index — is added and the result is re-normalised.
        """
        color_idx = indices[:, -1].long() % 3                              # (A,)
        base = self._color_features[color_idx.cpu()].to(indices.device)   # (A, D)

        if self.embedding_noise_std <= 0.0:
            return base

        # Polynomial hash of each index row → bank slot in [0, BANK_SIZE).
        primes = torch.tensor([1_000_003, 2_000_003, 3_000_003],
                              dtype=torch.long, device=indices.device)
        row_hashes = (indices.long() * primes[:indices.shape[1]]).sum(dim=-1)  # (A,)
        bank_idx   = row_hashes.abs() % self._BANK_SIZE                        # (A,)
        noise      = self._noise_bank[bank_idx.cpu()].to(indices.device)       # (A, D)

        perturbed = base + self.embedding_noise_std * noise   # (A, D)
        return F.normalize(perturbed, dim=-1)                 # (A, D)

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        A, T   = pushed.shape[0], self.n_augmentation_trials
        n      = self.image_size
        device = pushed.device

        # Direct reshape: pushed (A, n², 3) → (A, 3, n, n)
        grid = pushed.reshape(A, n, n, 3).permute(0, 3, 1, 2)  # (A, 3, n, n)

        # Range penalty: push values into [-1, 1].
        range_penalty = (F.relu(-1 - grid) + F.relu(grid - 1)).mean(dim=(1, 2, 3))  # (A,)

        # Clamp to [-1, 1] then rescale to [0, 1] for CLIP.
        grid_clipped = (grid.clamp(-1, 1) + 1) / 2  # (A, 3, n, n)

        # T independent augmented views with random background compositing.
        def _one_view(g):
            bg = self._random_bg(g)
            return self._augment(self.bg_alpha * g + (1 - self.bg_alpha) * bg)

        views = torch.cat([_one_view(grid_clipped) for _ in range(T)], dim=0)  # (A*T, 3, S, S)
        views = self._clip_normalise(views)

        self._model.to(device)
        img_features  = F.normalize(self._model.encode_image(views), dim=-1)  # (A*T, D)
        text_features = self._atom_text_features(indices).repeat(T, 1)        # (A*T, D)

        sim = (img_features * text_features).sum(dim=-1)          # (A*T,)
        clip_energy = -sim.reshape(T, A).mean(dim=0)              # (A,)
        return clip_energy + self.range_penalty_weight * range_penalty

class ImageTargetRegularizer(PushforwardRegularizer):
    """Penalises each pushed-forward atom for differing from a target image.

    The energy combines two terms:

    1. **MSE** — pixel-level L2 distance between the atom's grid image and the
       target (both min-max normalised to [0, 1]).
    2. **Perceptual** — L2 distance in the feature space of a frozen pretrained
       MobileNetV3-Small (ImageNet weights).

        E_a = mse_weight · MSE(grid_a, target)
            + perceptual_weight · ‖feat(grid_a) − feat(target)‖²

    Atoms are evaluated on a stratified ``image_size × image_size`` pixel grid
    built internally; no KNN interpolation is needed.

    Args:
        target_image_path:     Path to the target image file (loaded with PIL).
        image_size:            Grid resolution and target resize resolution.
        add_noise:             If True, adds per-epoch jitter to the stratified grid.
        mse_weight:            Weight for the MSE term (default 1.0).
        perceptual_weight:     Weight for the perceptual feature term (default 1.0).
        feature_upsample_size: Resolution fed to the feature extractor, or None to
                               use the grid image at native resolution (default 64).
        noise_std:             Std of Gaussian noise added per trial.
        n_noise_trials:        Number of noised copies per atom per step.
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        target_image_path: str,
        image_size: int = 32,
        add_noise: bool = False,
        mse_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        feature_upsample_size: int | None = 64,
        noise_std: float = 0.05,
        n_noise_trials: int = 8,
    ):
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image
        from infidictionary.domain_samplers import SquareSampler

        super().__init__(SquareSampler(stratified=True, add_noise=add_noise), image_size)
        self.image_size            = image_size
        self.mse_weight            = mse_weight
        self.perceptual_weight     = perceptual_weight
        self.feature_upsample_size = feature_upsample_size
        self.noise_std             = noise_std
        self.n_noise_trials        = n_noise_trials

        # Load and resize the target image to [0, 1] (3, S, S)
        img = Image.open(target_image_path).convert("RGB")
        transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self._target: torch.Tensor = transform(img)  # (3, S, S)

        # Frozen MobileNetV3-Small as feature extractor.
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self._feature_extractor = torch.nn.Sequential(
            backbone.features,
            backbone.avgpool,
        )
        self._feature_extractor.eval()
        for p in self._feature_extractor.parameters():
            p.requires_grad_(False)

        # Precompute target features once at init time
        with torch.no_grad():
            self._target_features: torch.Tensor = (
                self._feature_extractor(
                    self._imagenet_normalise(self._maybe_upsample(self._target.unsqueeze(0)))
                ).reshape(-1)  # (576,)
            )

        print(f"ImageTargetRegularizer: '{target_image_path}' → {image_size}×{image_size}, "
              f"feature_upsample_size={feature_upsample_size}, "
              f"mse_weight={mse_weight}, perceptual_weight={perceptual_weight}")

    def _imagenet_normalise(self, batch: torch.Tensor) -> torch.Tensor:
        device = batch.device
        mean = self._IMAGENET_MEAN.to(device)[None, :, None, None]
        std  = self._IMAGENET_STD.to(device)[None,  :, None, None]
        return (batch - mean) / std

    def _maybe_upsample(self, batch: torch.Tensor) -> torch.Tensor:
        if self.feature_upsample_size is None:
            return batch
        return F.interpolate(batch, size=self.feature_upsample_size,
                             mode="bilinear", align_corners=False)

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        A, T   = pushed.shape[0], self.n_noise_trials
        n      = self.image_size
        device = pushed.device

        # Direct reshape: pushed (A, n², 3) → (A, 3, n, n)
        grid = pushed.reshape(A, n, n, 3).permute(0, 3, 1, 2)  # (A, 3, n, n)

        # Range penalty on raw grid; clamp to [-1,1] then rescale to [0,1].
        range_penalty = (F.relu(-1 - grid) + F.relu(grid - 1)).mean(dim=(1, 2, 3))  # (A,)
        grid_clipped = (grid.clamp(-1, 1) + 1) / 2                # (A, 3, n, n)

        # Tile T copies per atom and add independent Gaussian noise to each
        grid_clipped = grid_clipped.repeat_interleave(T, dim=0)    # (A*T, 3, n, n)
        grid_clipped = grid_clipped + torch.randn_like(grid_clipped) * self.noise_std

        target = self._target.to(device)[None]                      # (1, 3, n, n)

        # ── MSE term ──────────────────────────────────────────────────────────
        mse = ((grid_clipped - target) ** 2).mean(dim=(1, 2, 3))   # (A*T,)
        mse = mse.reshape(A, T).mean(dim=1)                         # (A,)

        if self.perceptual_weight > 0:
            # ── Perceptual term ───────────────────────────────────────────────
            self._feature_extractor.to(device)
            atom_feats = self._feature_extractor(
                self._imagenet_normalise(self._maybe_upsample(grid_clipped))
            ).reshape(A * T, -1)                                     # (A*T, F)

            tgt_feats  = self._target_features.to(device)[None]     # (1, F)
            perceptual = ((atom_feats - tgt_feats) ** 2).mean(dim=-1)   # (A*T,)
            perceptual = perceptual.reshape(A, T).mean(dim=1)       # (A,)
        else:
            perceptual = 0

        return self.mse_weight * mse + self.perceptual_weight * perceptual + range_penalty
