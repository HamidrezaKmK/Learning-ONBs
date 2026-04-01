"""concept.py — Concept alignment losses for concept_basis training.

Two strategies are provided:

  SDSLoss  — Score Distillation Sampling (DreamFusion-style): a frozen
             text-conditioned diffusion model acts as a score function that
             steers rendered images toward a text concept.

  CLIPLoss — CLIP cosine-similarity: augmented views of rendered images are
             compared against a CLIP-encoded text embedding.  Inspired by
             Dream Fields (Jain et al., 2022).

Both expose the same two-method interface::

    text_encoded = loss.encode_text(prompt, device)   # called once before training
    scalar       = loss(images, text_encoded)          # called every gradient step
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class ConceptLoss(ABC):
    """Abstract base class for concept alignment losses.

    Subclasses must implement :meth:`encode_text` and :meth:`__call__`.

    The intended usage pattern is::

        loss = MyConceptLoss(...)
        text_encoded = loss.encode_text(prompt, device)   # once, before training
        for ...:
            images = render(...)        # (B, 3, H, W) in [-1, 1]
            l = loss(images, text_encoded)
            l.backward()
    """

    @abstractmethod
    def encode_text(self, prompt: str, device: torch.device) -> object:
        """Encode *prompt* into whatever representation this loss expects.

        The return value is opaque to callers — it is simply stored and later
        passed back to :meth:`__call__` unchanged.
        """

    @abstractmethod
    def __call__(self, image: torch.Tensor, text_encoded: object) -> torch.Tensor:
        """Compute the concept alignment loss for a batch of rendered images.

        Args:
            image:        ``(B, 3, H, W)`` batch of rendered images in ``[-1, 1]``.
            text_encoded: The value previously returned by :meth:`encode_text`.

        Returns:
            Scalar loss tensor (``requires_grad=True`` w.r.t. the contents of
            ``image``).
        """


# ─────────────────────────────────────────────────────────────────────────────
# SDSLoss
# ─────────────────────────────────────────────────────────────────────────────

class SDSLoss(ConceptLoss):
    """Score Distillation Sampling (SDS) loss using a pretrained text-conditioned
    diffusion model (Stable Diffusion v1-compatible).

    Background
    ----------
    DreamFusion (Poole et al., 2022) shows that a rendered image x can be
    guided toward a text concept y by treating a frozen text-to-image diffusion
    model as a *score function* on the space of images.  Concretely, for a
    given noise level t the SDS gradient is:

        ∇_θ L_SDS  =  E_t[ w(t) · (ε̂(z_t, t, y) - ε) · ∂z/∂θ ]

    where:
      • z  = E(x) is the VAE-encoded latent of the rendered image (differentiable).
      • ε  ~ N(0, I) is the noise used to corrupt z to z_t.
      • ε̂  = UNet(z_t, t, text_embeds) is the noise *predicted* by the frozen
              UNet with classifier-free guidance.
      • w(t) = σ_t² is a weighting term.

    Crucially, the gradient does NOT flow through the UNet itself (it is frozen
    and detached); it flows only through z → x → neural_isometry parameters.

    Implementation
    --------------
    We implement this via a *pseudo-loss* whose gradient w.r.t. z equals the
    SDS direction:

        loss  =  (sds_grad.detach() · z).sum()
              →  ∂loss/∂z  =  sds_grad  =  w(t) · (ε̂_cfg - ε)

    Args:
        model_id:       HuggingFace model ID for a Stable Diffusion v1-compatible
                        model (default "runwayml/stable-diffusion-v1-5").
        guidance_scale: Classifier-free guidance scale (default 7.5).
        t_range:        Fraction of the total timestep schedule from which to
                        uniformly sample t, expressed as (t_min_frac, t_max_frac).
                        Default (0.02, 0.98) avoids the extreme ends of the schedule.
        render_size:    Spatial resolution expected by the VAE encoder.  Rendered
                        images smaller than this are bilinearly upsampled (default 512).
        mode:           ``"sds"`` (pseudo-loss, UNet frozen, cheap) or ``"direct"``
                        (weighted MSE through UNet Jacobian, more expensive).
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        guidance_scale: float = 7.5,
        t_range: tuple[float, float] = (0.02, 0.98),
        render_size: int = 512,
        mode: str = "sds",
        n_augmentation_trials: int = 1,
    ):
        from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer
        from transformers import logging as hf_transformers_logging
        from torchvision.transforms import v2 as T

        assert mode in ("sds", "direct"), f"mode must be 'sds' or 'direct', got '{mode}'"
        self.mode = mode
        self.guidance_scale = guidance_scale
        self.render_size = render_size

        # Random crop + flip applied before VAE encoding.  Exposing the model
        # to differently-cropped views at each step prevents SD from locking
        # onto a single canonical orientation and position.
        
        self.n_augmentation_trials = n_augmentation_trials
        if n_augmentation_trials == 0:
            # Bilinear upsample path — handled in __call__ when T_aug == 0.
            self._sds_augment = None
        else:
            self._sds_augment = T.Compose([
                T.RandomResizedCrop(render_size, scale=(0.4, 1.0), ratio=(0.75, 4 / 3)),
                T.RandomHorizontalFlip(p=0.5),
            ])

        # Both suppressions are for library-internal issues unrelated to our code:
        #   - "local_dir_use_symlinks" is a deprecated arg huggingface_hub passes to itself.
        #   - transformers logs an "UNEXPECTED" load report for position_ids because the
        #     SD v1.5 checkpoint stores it as a persistent buffer but newer transformers
        #     dropped it; the key is harmless and the model behaves identically without it.
        _prev_verbosity = hf_transformers_logging.get_verbosity()
        hf_transformers_logging.set_verbosity_error()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")

            self.tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            self.text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
            self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
            self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
            self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        hf_transformers_logging.set_verbosity(_prev_verbosity)

        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad_(False)

        T = self.noise_scheduler.config.num_train_timesteps
        self.t_min = max(1, int(t_range[0] * T))
        self.t_max = min(T - 1, int(t_range[1] * T))

        # Cache alphas_cumprod on CPU; will be moved to device in __call__.
        self._alphas_cumprod: torch.Tensor = self.noise_scheduler.alphas_cumprod.clone()

        print(self.t_min, self.t_max, "*************")
        print(
            f"SDSLoss: model={model_id}, mode={mode}, guidance_scale={guidance_scale}, "
            f"t_range=({self.t_min}, {self.t_max}), render_size={render_size}, "
            f"n_augmentation_trials={n_augmentation_trials}"
        )

    def encode_text(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Encode the prompt and the empty string for classifier-free guidance.

        Returns:
            text_embeds: ``(2, seq_len, D)`` — stacked ``[uncond, cond]`` ready for CFG.
        """
        self.text_encoder.to(device)
        tokens = self.tokenizer(
            ["", prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = self.text_encoder(tokens.input_ids.to(device)).last_hidden_state
        return embeds  # (2, seq_len, D)

    def __call__(
        self,
        image: torch.Tensor,
        text_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the diffusion guidance loss for a batch of rendered images.

        Args:
            image:        ``(B, 3, H, W)`` batch of rendered images in ``[-1, 1]``.
            text_encoded: ``(2, seq_len, D)`` — ``[uncond, cond]`` — from :meth:`encode_text`.

        Returns:
            Scalar loss (sds: pseudo-loss; direct: weighted MSE).
        """
        device = image.device
        B = image.shape[0]
        T_aug = self.n_augmentation_trials
        self.vae.to(device)
        self.unet.to(device)

        # ── Build views ───────────────────────────────────────────────────────
        # n_augmentation_trials == 0: no augmentation — bilinear upsample to
        # render_size, matching the original DreamFusion preprocessing.
        # n_augmentation_trials >= 1: T_aug independently cropped + flipped views
        # per image, exposing SD to different crops at each step.
        img_clamped = image.clamp(-1.0, 1.0)
        if T_aug == 0:
            if img_clamped.shape[-1] != self.render_size or img_clamped.shape[-2] != self.render_size:
                views = F.interpolate(
                    img_clamped, size=self.render_size, mode="bilinear", align_corners=False
                )
            else:
                views = img_clamped
        else:
            views = torch.cat(
                [self._sds_augment(img_clamped) for _ in range(T_aug)], dim=0
            )  # (B * T_aug, 3, render_size, render_size)
        with torch.no_grad():
            B_eff = views.shape[0]

            # ── Encode the batch to latents z ─────────────────────────────────────
            # scaling_factor (0.18215 for SD v1-5) is mandatory: the DDPM noise
            # schedule, UNet weights, and alphas_cumprod are all calibrated for
            # this specific latent scale.  Omitting it makes z ~5.5× too large,
            # which produces completely wrong UNet noise predictions.
            z = self.vae.encode(views).latent_dist.mode() * self.vae.config.scaling_factor
            # z: (B_eff, 4, render_size/8, render_size/8),  requires_grad=True

            # ── Sample one random timestep per view and add noise ─────────────────
            alphas  = self._alphas_cumprod.to(device)
            t_idx   = torch.randint(self.t_min, self.t_max, (B_eff,), device=device)
            noise   = torch.randn_like(z)

            # print(alphas[t_idx], "\n*******")
            alpha_t = alphas[t_idx].sqrt().view(B_eff, 1, 1, 1)
            sigma_t = (1.0 - alphas[t_idx]).sqrt().view(B_eff, 1, 1, 1)
            # print(z.std(), z.mean(), sigma_t.mean(), "\nSTATS")
            z_t     = noise * sigma_t +  alpha_t * z #  + 

            # ── Build batched UNet inputs for CFG ─────────────────────────────────
            w_t     = sigma_t ** 2
            z_t_in  = z_t.repeat(2, 1, 1, 1)
            t_in    = t_idx.repeat(2)
            text_in = torch.cat([
                text_encoded[:1].expand(B_eff, -1, -1),
                text_encoded[1:].expand(B_eff, -1, -1),
            ], dim=0)

            # if self.mode == "sds":
            #     with torch.no_grad():
            #         noise_pred   = self.unet(z_t_in, t_in, encoder_hidden_states=text_in).sample
            #         noise_uncond, noise_cond = noise_pred[:B_eff], noise_pred[B_eff:]
            #         noise_cfg    = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)

            #     sds_grad = w_t * (noise_cfg - noise)
            #     # nan_to_num: UNet can produce NaN predictions for out-of-distribution
            #     # latents early in training; without this the NaN propagates all the
            #     # way back through the neural isometry (dreamfusionacc does the same).
            #     sds_grad = torch.nan_to_num(sds_grad).detach()
            #     # Pseudo-loss: ∂loss/∂z = sds_grad.  Dividing by B_eff keeps the
            #     # per-image gradient scale constant regardless of n_augmentation_trials.
            #     loss = (sds_grad * z).sum() / B_eff

            # else:  # "direct"
            noise_pred   = self.unet(z_t_in, t_in, encoder_hidden_states=text_in).sample
            noise_uncond, noise_cond = noise_pred[:B_eff], noise_pred[B_eff:]
            noise_cfg    = noise_uncond + self.guidance_scale * (noise_cond - noise_uncond)
            z0_pred = (z_t - sigma_t * noise_cfg) / alpha_t
        # loss = (w_t * (z - z0_pred.detach()) ** 2).mean()
            decoder_output = self.vae.decode(z0_pred / self.vae.config.scaling_factor)
            # breakpoint()
            x0_pred = decoder_output.sample.detach()
        loss = (w_t * (views - x0_pred) ** 2).mean() # * 
        return loss, x0_pred, self.vae.decode(z_t / self.vae.config.scaling_factor).sample.detach()


# ─────────────────────────────────────────────────────────────────────────────
# CLIPLoss
# ─────────────────────────────────────────────────────────────────────────────

class CLIPLoss(ConceptLoss):
    """CLIP cosine-similarity loss for concept alignment.

    For each forward pass, ``n_augmentation_trials`` independently augmented
    views of every rendered image are encoded with a frozen OpenCLIP model and
    compared against the pre-encoded text embedding.  The loss is the negative
    mean cosine similarity.  Out-of-range pixel values are handled by the
    external ``range_penalty`` in ``concept_basis_training``; this class only
    clamps before augmentation so CLIP sees valid inputs.

    Augmentations (random crop, flip, colour jitter, random erasing) and
    structured random backgrounds (Dream Fields) force the model to respond to
    semantically meaningful spatial structure rather than surface statistics.

    Args:
        model_name:            OpenCLIP model architecture (default ``"ViT-B-32"``).
        pretrained:            OpenCLIP pretrained weights tag (default ``"openai"``).
        n_augmentation_trials: Independently augmented views per image per step.
        bg_alpha:              Foreground opacity when compositing over a random
                               background (default 0.8; 1.0 disables compositing).
    """

    _CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
    _CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        n_augmentation_trials: int = 8,
        bg_alpha: float = 0.8,
    ):
        import open_clip
        from torchvision.transforms import v2 as T

        self.n_augmentation_trials = n_augmentation_trials
        self.bg_alpha = bg_alpha

        self._model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)

        self._clip_size: int = self._model.visual.image_size
        if isinstance(self._clip_size, (list, tuple)):
            self._clip_size = self._clip_size[0]

        self._tokenizer = open_clip.get_tokenizer(model_name)

        self._augment = T.Compose([
            T.RandomResizedCrop(self._clip_size, scale=(0.25, 1.0), ratio=(0.5, 2.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

        print(
            f"CLIPLoss: model={model_name}/{pretrained}, clip_size={self._clip_size}, "
            f"n_augmentation_trials={n_augmentation_trials}, bg_alpha={bg_alpha}"
        )

    def encode_text(self, prompt: str, device: torch.device) -> torch.Tensor:
        """Encode a text prompt to a normalised CLIP feature vector.

        Returns:
            text_features: ``(D,)`` unit-norm CLIP text embedding.
        """
        self._model.to(device)
        tokens = self._tokenizer([prompt])
        with torch.no_grad():
            feats = F.normalize(
                self._model.encode_text(tokens.to(device)), dim=-1
            )
        return feats.squeeze(0)  # (D,)

    def _random_bg(self, g: torch.Tensor) -> torch.Tensor:
        """Random structured background of the same spatial shape as *g*.

        Randomly chooses between uniform noise and a Fourier sinusoidal
        texture.  Structured backgrounds (Dream Fields finding) prevent CLIP
        from latching onto background colour alone.
        """
        B, C, H, W = g.shape
        device = g.device
        if torch.randint(0, 2, (1,)).item() == 0:
            return torch.rand_like(g)
        freq = torch.randint(2, 6, (B, C, 1, 1), device=device).float()
        ph_x = torch.rand(B, C, 1, 1, device=device) * 2 * math.pi
        ph_y = torch.rand(B, C, 1, 1, device=device) * 2 * math.pi
        xs = torch.linspace(0, 2 * math.pi, W, device=device)[None, None, None, :]
        ys = torch.linspace(0, 2 * math.pi, H, device=device)[None, None, :, None]
        return (0.5 + 0.5 * torch.sin(freq * xs + ph_x) * torch.sin(freq * ys + ph_y)).clamp(0, 1)

    def _clip_normalise(self, batch: torch.Tensor) -> torch.Tensor:
        device = batch.device
        mean = self._CLIP_MEAN.to(device)[None, :, None, None]
        std  = self._CLIP_STD.to(device)[None,  :, None, None]
        return (batch - mean) / std

    def __call__(
        self,
        image: torch.Tensor,
        text_encoded: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CLIP alignment loss for a batch of rendered images.

        Args:
            image:        ``(B, 3, H, W)`` batch of rendered images (may exceed ``[-1, 1]``).
            text_encoded: ``(D,)`` normalised text embedding from :meth:`encode_text`.

        Returns:
            Scalar loss: mean negative CLIP cosine similarity.
        """
        B, T = image.shape[0], self.n_augmentation_trials
        device = image.device

        # Clamp to [-1, 1] then rescale to [0, 1] for CLIP.
        # Out-of-range values contribute zero gradient here; the external
        # range_penalty in concept_basis_training supplies that gradient.
        img_clipped = (image.clamp(-1, 1) + 1) / 2  # (B, 3, H, W)

        # T independently augmented views with random background compositing.
        def _one_view(g: torch.Tensor) -> torch.Tensor:
            bg = self._random_bg(g)
            augmented = self._augment(self.bg_alpha * g + (1 - self.bg_alpha) * bg)
            return augmented.clamp(0.0, 1.0)

        views = torch.cat([_one_view(img_clipped) for _ in range(T)], dim=0)  # (B*T, 3, S, S)
        views = self._clip_normalise(views)

        self._model.to(device)
        img_features  = F.normalize(self._model.encode_image(views), dim=-1)   # (B*T, D)
        text_features = text_encoded.to(device).unsqueeze(0).expand(B * T, -1) # (B*T, D)

        sim = (img_features * text_features).sum(dim=-1)  # (B*T,)
        clip_energy = -sim.reshape(T, B).mean(dim=0)      # (B,)

        return clip_energy.mean()
