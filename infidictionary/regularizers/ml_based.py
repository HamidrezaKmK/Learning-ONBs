
from .base import PushforwardRegularizer
import torch
import torch.nn.functional as F


def build_grid_weights(
    tgt_coords: torch.Tensor, k_nn: int = 8, image_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build KNN interpolation weights from pixel centres to irregular coordinates.

    Args:
        tgt_coords:  (N, d) coordinates; the first two dimensions are used.
        k_nn:        Number of nearest quadrature points per pixel centre.
        image_size:  Side length of the square output grid (default 32).

    Returns:
        nn_idx: (image_size², k) LongTensor — indices into the N input points.
        nn_w:   (image_size², k) FloatTensor — normalised inverse-distance weights.
    """
    import faiss

    dtype   = tgt_coords.dtype
    spatial = tgt_coords[:, :2].detach().cpu().float().contiguous().numpy()  # (N, 2)

    t = (torch.arange(image_size, dtype=dtype) + 0.5) / image_size
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    queries = torch.stack([gx, gy], dim=-1).reshape(-1, 2).float().numpy()  # (S², 2)

    k = min(k_nn, spatial.shape[0])
    index = faiss.IndexFlatL2(2)
    index.add(spatial)
    sq_dists, nn_idx = index.search(queries, k)  # (S², k) each

    nn_w = 1.0 / (torch.from_numpy(sq_dists).to(dtype).sqrt() + 1e-8)
    nn_w = nn_w / nn_w.sum(dim=1, keepdim=True)

    return torch.from_numpy(nn_idx).long(), nn_w


def atoms_to_grid_images(
    pushed: torch.Tensor,
    nn_idx: torch.Tensor,
    nn_w: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Interpolate scattered atom values onto a square grid using precomputed KNN weights.

    Args:
        pushed:     (A, N, C) atom values at N irregular coordinates.
        nn_idx:     (image_size², k) nearest-neighbour indices from :func:`build_grid_weights`.
        nn_w:       (image_size², k) interpolation weights from :func:`build_grid_weights`.
        image_size: Side length of the square output grid; must match what was
                    passed to :func:`build_grid_weights`.

    Returns:
        grid: (A, C, image_size, image_size) images.
    """
    A, N, C  = pushed.shape
    n_pixels = image_size * image_size
    device   = pushed.device
    nn_idx   = nn_idx.to(device)
    nn_w     = nn_w.to(device)
    k        = nn_idx.shape[1]

    vals      = pushed[:, nn_idx.reshape(-1), :].reshape(A, n_pixels, k, C)
    grid_flat = (vals * nn_w[None, :, :, None]).sum(dim=2)                        # (A, S², C)
    return grid_flat.reshape(A, image_size, image_size, C).permute(0, 3, 1, 2)   # (A, C, S, S)


class ClassActivationRegularizer(PushforwardRegularizer):
    """Maximises a target CIFAR-10 class logit for each pushed-forward atom.

    In update_coordinates, a FAISS KNN graph maps each 32×32 pixel centre to
    its k nearest quadrature points with normalised inverse-distance weights.
    In _energy_from_atoms those weights convert the (A, N, 3) atom values to a
    (A, 3, 32, 32) image, which is passed through a frozen CIFAR-10 classifier.
    Returning the negative target-class logit as energy makes minimisation
    equivalent to maximising that class score.

    CIFAR-10 indices: 0 airplane 1 automobile 2 bird 3 cat 4 deer
                      5 dog 6 frog 7 horse 8 ship 9 truck

    Args:
        target_class: CIFAR-10 class to maximise (default 3 = cat).
        k_nn:         Nearest quadrature points per pixel (default 8).
        image_size:   Resolution of the grid fed to the classifier (default 32).
        hub_repo:     torch.hub source for the CIFAR-10 model.
        hub_model:    Model name within that repo.
    """

    CIFAR10_CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    _MEAN = torch.tensor([0.4914, 0.4822, 0.4465])
    _STD  = torch.tensor([0.2023, 0.1994, 0.2010])

    def __init__(
        self,
        target_class: int = 3,
        k_nn: int = 8,
        image_size: int = 32,
        hub_repo: str = "chenyaofo/pytorch-cifar-models",
        hub_model: str = "cifar10_resnet20",
        noise_std: float = 0.05,
        n_noise_trials: int = 8,
    ):
        assert 0 <= target_class <= 9
        self.target_class   = target_class
        self.k_nn           = k_nn
        self.image_size     = image_size
        self.noise_std      = noise_std
        self.n_noise_trials = n_noise_trials
        self.model          = torch.hub.load(hub_repo, hub_model, pretrained=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._nn_idx: torch.Tensor | None = None
        self._nn_w:   torch.Tensor | None = None
        print(f"ClassActivationRegularizer: target='{self.CIFAR10_CLASSES[target_class]}' "
              f"(class {target_class}), model={hub_model}, "
              f"noise_std={noise_std}, n_noise_trials={n_noise_trials}")

    def update_coordinates(self, coords: torch.Tensor) -> None:
        """Build FAISS KNN from pixel centres to quadrature points."""
        self._nn_idx, self._nn_w = build_grid_weights(coords, self.k_nn, self.image_size)

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        if self._nn_idx is None:
            raise RuntimeError("update_coordinates must be called before compute_energy.")

        A, T   = pushed.shape[0], self.n_noise_trials
        device = pushed.device
        grid   = atoms_to_grid_images(pushed, self._nn_idx, self._nn_w, self.image_size)

        # Min-max normalise to [0, 1]
        vmin = grid.flatten(1).min(1).values[:, None, None, None]
        vmax = grid.flatten(1).max(1).values[:, None, None, None]
        grid = (grid - vmin) / (vmax - vmin + 1e-8)             # (A, 3, S, S)

        # Tile T copies per atom and add independent Gaussian noise to each
        grid = grid.repeat_interleave(T, dim=0)                  # (A*T, 3, S, S)
        grid = (grid + torch.randn_like(grid) * self.noise_std).clamp(0.0, 1.0)

        mean  = self._MEAN.to(device)[None, :, None, None]
        std   = self._STD.to(device)[None, :, None, None]
        batch = (grid - mean) / std

        self.model.to(device)
        logits = self.model(batch)                               # (A*T, 10)
        return -logits[:, self.target_class].reshape(A, T).mean(dim=1)  # (A,)


class CLIPRegularizer(PushforwardRegularizer):
    """Maximises CLIP cosine similarity with a per-atom text prompt derived from
    the atom's multi-index.

    The prompt template is "A sitting {fur} {subject} colored in {color}":

      - **color** is determined by the last index component modulo 3:
        0 → Red, 1 → Green, 2 → Blue.
      - **fur**   is determined by the Chebyshev frequency radius
        max(|k₁|, |k₂|, ...) of all index components except the last.
        Low-radius atoms encode global/smooth content; high-radius atoms
        encode fine detail — analogous to how much visible fur texture a cat
        image would need to represent that atom faithfully:
            radius < t0               → "Hairless"
            t0 ≤ radius < t1          → "Short-haired"
            t1 ≤ radius               → "Fluffy"
        (t2 acts as a soft ceiling; radii ≥ t2 also map to "Fluffy".)

    All 9 (fur × color) text embeddings are precomputed once at init and stored
    on CPU. At each step atom images are splatted to a grid, min-max normalised,
    then passed through T independent random spatial augmentations before a
    single batched CLIP forward pass.

    Args:
        subject:               Subject noun in the prompt template (default "Cat").
        k_nn:                  Nearest quadrature points per pixel (default 8).
        image_size:            Resolution of the intermediate KNN grid (default 32).
        model_name:            OpenCLIP model architecture (default "ViT-B-32").
        pretrained:            OpenCLIP pretrained weights tag (default "openai").
        n_augmentation_trials: Independently augmented views per atom per step
                               (default 8).
        hair_thresholds:       Three Chebyshev-radius boundaries that gate
                               Hairless / Short-haired / Fluffy (default (1, 2, 3)).
    """

    _CLIP_MEAN  = torch.tensor([0.48145466, 0.4578275,  0.40821073])
    _CLIP_STD   = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    _FUR_LEVELS = ["Hairless", "Short-haired", "Fluffy"]
    _COLORS     = ["Red", "Green", "Blue"]

    def __init__(
        self,
        subject: str = "Cat",
        k_nn: int = 8,
        image_size: int = 32,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        n_augmentation_trials: int = 8,
        hair_thresholds: tuple[float, float, float] = (1.0, 2.0, 3.0),
        bg_alpha: float = 0.8,
    ):
        import open_clip
        from torchvision.transforms import v2 as T

        self.subject               = subject
        self.k_nn                  = k_nn
        self.image_size            = image_size
        self.n_augmentation_trials = n_augmentation_trials
        self.hair_thresholds       = hair_thresholds
        self.bg_alpha              = bg_alpha
        self._nn_idx: torch.Tensor | None = None
        self._nn_w:   torch.Tensor | None = None

        self._model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._clip_size: int = self._model.visual.image_size
        if isinstance(self._clip_size, (list, tuple)):
            self._clip_size = self._clip_size[0]

        # Precompute all 9 (fur × color) text embeddings and store on CPU.
        tokenizer = open_clip.get_tokenizer(model_name)
        prompts = [
            f"A sitting {fur} {subject} colored in {color}"
            for fur in self._FUR_LEVELS
            for color in self._COLORS
        ]
        with torch.no_grad():
            feats = F.normalize(self._model.encode_text(tokenizer(prompts)), dim=-1)  # (9, D)
        self._prompt_features = feats.reshape(3, 3, -1).cpu()  # (fur, color, D)

        self._augment = T.Compose([
            T.RandomResizedCrop(self._clip_size, scale=(0.5, 1.0), ratio=(0.75, 4.0 / 3.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        ])

        print(f"CLIPRegularizer: subject='{subject}', model={model_name}/{pretrained}, "
              f"clip_size={self._clip_size}, n_augmentation_trials={n_augmentation_trials}, "
              f"hair_thresholds={hair_thresholds}")
        for i, prompt in enumerate(prompts):
            print(f"  [{self._FUR_LEVELS[i // 3]}, {self._COLORS[i % 3]}] {prompt}")

    def update_coordinates(self, coords: torch.Tensor) -> None:
        self._nn_idx, self._nn_w = build_grid_weights(coords, self.k_nn, self.image_size)

    def _clip_normalise(self, batch: torch.Tensor) -> torch.Tensor:
        device = batch.device
        mean = self._CLIP_MEAN.to(device)[None, :, None, None]
        std  = self._CLIP_STD.to(device)[None,  :, None, None]
        return (batch - mean) / std

    def _atom_text_features(self, indices: torch.Tensor) -> torch.Tensor:
        """Return (A, D) text features for a batch of multi-indices.

        Args:
            indices: (A, d_idx) integer multi-indices — last component is channel,
                     remaining components are frequency coordinates.
        """
        color_idx = indices[:, -1].long() % 3                                  # (A,)

        # Chebyshev radius: max absolute frequency component (excluding channel)
        freq_radius = indices[:, :-1].abs().float().max(dim=-1).values          # (A,)

        # 3 thresholds → up to 4 buckets; clamp to keep within 3 fur levels
        boundaries = torch.tensor(list(self.hair_thresholds),
                                  dtype=freq_radius.dtype, device=freq_radius.device)
        fur_idx = torch.bucketize(freq_radius, boundaries).clamp(max=2)         # (A,)

        return self._prompt_features[fur_idx.cpu(), color_idx.cpu()].to(indices.device)  # (A, D)

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        if self._nn_idx is None:
            raise RuntimeError("update_coordinates must be called before compute_energy.")

        A, T   = pushed.shape[0], self.n_augmentation_trials
        device = pushed.device

        grid = atoms_to_grid_images(pushed, self._nn_idx, self._nn_w, self.image_size)
        vmin = grid.flatten(1).min(1).values[:, None, None, None]
        vmax = grid.flatten(1).max(1).values[:, None, None, None]
        grid = (grid - vmin) / (vmax - vmin + 1e-8)    # (A, 3, S, S) in [0, 1]

        # T independent augmented views; torch.cat along atom dim → (A*T, ...)
        # Ordering: [a0t0, a1t0, ..., a(A-1)t0, a0t1, ..., a(A-1)t(T-1)]
        # Each trial composites over a fresh random background before augmenting,
        # forcing CLIP to respond to spatial cat structure rather than colour alone.
        def _one_view(g):
            bg = torch.rand_like(g)
            return self._augment(self.bg_alpha * g + (1 - self.bg_alpha) * bg)

        views = torch.cat([_one_view(grid) for _ in range(T)], dim=0)
        views = self._clip_normalise(views)

        self._model.to(device)
        img_features  = F.normalize(self._model.encode_image(views), dim=-1)  # (A*T, D)
        text_features = self._atom_text_features(indices).repeat(T, 1)        # (A*T, D)

        sim = (img_features * text_features).sum(dim=-1)  # (A*T,)
        return -sim.reshape(T, A).mean(dim=0)             # (A,)


class ImageTargetRegularizer(PushforwardRegularizer):
    """Penalises each pushed-forward atom for differing from a target image.

    The energy combines two terms:

    1. **MSE** — pixel-level L2 distance between the atom's grid image and the
       target (both min-max normalised to [0, 1]).
    2. **Perceptual** — L2 distance in the feature space of a frozen pretrained
       MobileNetV3-Small (ImageNet weights), measuring semantic dissimilarity.
       The grid image is optionally upsampled to ``feature_upsample_size`` before
       the feature pass; set to ``None`` to pass the grid image directly without
       any upsampling.

        E_a = mse_weight · MSE(grid_a, target)
            + perceptual_weight · ‖feat(grid_a) − feat(target)‖²

    Minimising this energy pushes the atom images toward the target in both
    pixel and semantic space.

    Args:
        target_image_path:     Path to the target image file (loaded with PIL).
        image_size:            Side length of the comparison grid (default 32).
        k_nn:                  Nearest quadrature points per pixel (default 8).
        mse_weight:            Weight for the MSE term (default 1.0).
        perceptual_weight:     Weight for the perceptual feature term (default 1.0).
        feature_upsample_size: Resolution fed to the feature extractor, or None to
                               pass the grid image at its native resolution (default 64).
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        target_image_path: str,
        image_size: int = 32,
        k_nn: int = 8,
        mse_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        feature_upsample_size: int | None = 64,
        noise_std: float = 0.05,
        n_noise_trials: int = 8,
    ):
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image

        self.image_size            = image_size
        self.k_nn                  = k_nn
        self.mse_weight            = mse_weight
        self.perceptual_weight     = perceptual_weight
        self.feature_upsample_size = feature_upsample_size
        self.noise_std             = noise_std
        self.n_noise_trials        = n_noise_trials
        self._nn_idx: torch.Tensor | None = None
        self._nn_w:   torch.Tensor | None = None

        # Load and resize the target image to [0, 1] (3, S, S)
        img = Image.open(target_image_path).convert("RGB")
        transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor()])
        self._target: torch.Tensor = transform(img)  # (3, S, S)

        # Frozen MobileNetV3-Small as feature extractor.
        # Strip the classifier head; adaptive_avg_pool2d inside the backbone
        # produces a (B, 576, 1, 1) tensor which we flatten to (B, 576).
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
        """Apply ImageNet mean/std normalisation to a (B, 3, H, W) batch."""
        device = batch.device
        mean = self._IMAGENET_MEAN.to(device)[None, :, None, None]
        std  = self._IMAGENET_STD.to(device)[None,  :, None, None]
        return (batch - mean) / std

    def _maybe_upsample(self, batch: torch.Tensor) -> torch.Tensor:
        """Upsample to feature_upsample_size if set, otherwise return as-is."""
        if self.feature_upsample_size is None:
            return batch
        return F.interpolate(batch, size=self.feature_upsample_size,
                             mode="bilinear", align_corners=False)

    def update_coordinates(self, coords: torch.Tensor) -> None:
        self._nn_idx, self._nn_w = build_grid_weights(coords, self.k_nn, self.image_size)

    def _energy_from_atoms(self, tgt_coords, init, pushed, indices) -> torch.Tensor:
        if self._nn_idx is None:
            raise RuntimeError("update_coordinates must be called before compute_energy.")

        A, T   = pushed.shape[0], self.n_noise_trials
        device = pushed.device

        # Interpolate to (A, 3, S, S) and min-max normalise each atom to [0, 1]
        grid = atoms_to_grid_images(pushed, self._nn_idx, self._nn_w, self.image_size)
        vmin = grid.flatten(1).min(1).values[:, None, None, None]
        vmax = grid.flatten(1).max(1).values[:, None, None, None]
        grid = (grid - vmin) / (vmax - vmin + 1e-8)                # (A, 3, S, S)

        # Tile T copies per atom and add independent Gaussian noise to each
        grid = grid.repeat_interleave(T, dim=0)                     # (A*T, 3, S, S)
        grid = (grid + torch.randn_like(grid) * self.noise_std).clamp(0.0, 1.0)

        target = self._target.to(device)[None]                      # (1, 3, S, S)

        # ── MSE term ──────────────────────────────────────────────────────────
        mse = ((grid - target) ** 2).mean(dim=(1, 2, 3))            # (A*T,)
        mse = mse.reshape(A, T).mean(dim=1)                         # (A,)

        if self.perceptual_weight > 0:
            # ── Perceptual term ───────────────────────────────────────────────
            self._feature_extractor.to(device)
            atom_feats = self._feature_extractor(
                self._imagenet_normalise(self._maybe_upsample(grid))
            ).reshape(A * T, -1)                                     # (A*T, F)

            tgt_feats  = self._target_features.to(device)[None]     # (1, F)
            perceptual = ((atom_feats - tgt_feats) ** 2).mean(dim=-1)   # (A*T,)
            perceptual = perceptual.reshape(A, T).mean(dim=1)       # (A,)
        else:
            perceptual = 0

        return self.mse_weight * mse + self.perceptual_weight * perceptual
