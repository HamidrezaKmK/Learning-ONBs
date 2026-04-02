import datetime
import os

import matplotlib
matplotlib.use("Agg")
import hydra
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

from infidictionary.checkpointing import Checkpointer
from infidictionary.concept import ConceptLoss
from infidictionary.dictionaries.base import InfiDictionary
from infidictionary.domain_samplers import SquareSampler
from infidictionary.neural_isometries import NeuralIsometry
from infidictionary.neural_isometries import EulerianIsometry
from infidictionary.networks import NeuralField
from training_utils import get_grad_norm, get_param_norm, get_avg_lr, step_scheduler

OmegaConf.register_new_resolver("eval", eval)

# A stratified, noise-free grid sampler — used for rendering inside concept_basis.
# SquareSampler with indexing='ij' produces coords where:
#   coords[:, 0]  are x-values (vary slowly — outer loop → rows of the grid)
#   coords[:, 1]  are y-values (vary quickly — inner loop → columns of the grid)
# After push-forward, reshaping to (n, n, C) and permuting to (C, n, n) therefore
# gives a (C, H, W) image with H = x-axis, W = y-axis. This matches the convention
# already used by CLIPRegularizer and is internally consistent.
_RENDER_SAMPLER = SquareSampler(stratified=True, add_noise=False)



def concept_basis_training(
    neural_isometry: NeuralIsometry,
    initial_dictionary: InfiDictionary,
    concept_loss: ConceptLoss,
    text_embeds: torch.Tensor,
    n_epochs: int,
    device: torch.device,
    optim_isometry: torch.optim.Optimizer,
    scheduler_isometry: torch.optim.lr_scheduler._LRScheduler | None,
    K: int,
    coefficient_decay: float,
    image_size: int,
    wandb_enabled: bool,
    model_state_kwargs: dict,
    pushforward_kwargs: dict,
    checkpointer: Checkpointer | None,
    checkpoint: dict | None,
    batch_size: int,
    callbacks: list,
    mean_function: NeuralField | None = None,
    optim_mean_function: torch.optim.Optimizer | None = None,
    scheduler_mean_function: torch.optim.lr_scheduler._LRScheduler | None = None,
    range_penalty_weight: float = 1.0,
    max_grad_norm: float | None = None,
    grad_accumulation_steps: int = 1,
    variation_strength: float = 0.0,
):
    """Train Q_θ so that random K-sparse linear combinations of pushed-forward
    basis elements render like a target text concept under Score Distillation
    Sampling (DreamFusion-style).

    Each gradient step proceeds as follows (repeated ``grad_accumulation_steps``
    times before the optimizer step, for gradient accumulation):

      1. Fix the atom index set via get_truncated_indices(K) (same atoms every
         accumulation sub-step within an epoch; the grid is resampled each epoch).
      2. Sample ``batch_size × len(indices)`` coefficients grouped into a batch:
           coefficients ~ N(0, pmf(i_j)^{2·coefficient_decay})  ∈ R^{B × K}
         so that each of the B images in the mini-batch gets its own independent
         coefficient draw over the same K atoms.
      3. Sample target-domain coordinates on a regular grid, pull back (no-grad)
         to source-domain coordinates.
      4. Evaluate the initial atoms ψ_j = dict[i_j] at the source coordinates:  (K, N, C).
      5. Push forward all K atoms through Q_θ:  φ_j = Q_θ ψ_j  ∈ R^{K × N × C}.
      6. Form B linear combinations via einsum → (B, N, C), reshape to (B, C, H, W).
      7. Apply tanh, then compute the diffusion guidance loss over the whole batch.

    Args:
        K:                      Truncation level — uses the first (2K+1)² × C atoms.
        coefficient_decay:      Exponent α: std(c_j) = pmf(i_j)^α.
                                α=0 → equal std; α>0 → low-freq atoms get larger coefficients.
        batch_size:             Number of independently drawn coefficient sets (images)
                                per forward pass.
        grad_accumulation_steps: Number of forward/backward passes before an optimizer
                                step. Losses are divided by this value so the effective
                                gradient is the average over accumulation steps.
    """
    neural_isometry = neural_isometry.to(device)
    text_embeds = text_embeds.to(device)
    optim_isometry.zero_grad()
    if optim_mean_function is not None:
        optim_mean_function.zero_grad()

    start_epoch = 0
    if checkpoint is not None and checkpointer is not None:
        start_epoch, _ = checkpointer.restore(checkpoint)

    pbar = tqdm(range(n_epochs))
    for epoch_i in pbar:
        if epoch_i < start_epoch:
            continue


        # ── Sample target-domain quadrature grid and pull back to source domain ─
        # We always sample tgt_coords (the output/target domain of Q_θ) and
        # derive the corresponding src_coords via the pullback.  This is correct
        # for all isometry types:
        #   - EulerianIsometry: operates in function-value space only, so
        #     src_coords = tgt_coords (the coordinate map is the identity).
        #   - LagrangianIsometry: tgt_coords are the output domain; the pullback
        #     finds the pre-image in the source domain.
        # The pullback is computed once per step under no_grad and detached so
        # that no gradient flows through the coordinate transformation — gradients
        # flow only through the pushforward of atom values (step 5 below).
        tgt_coords = _RENDER_SAMPLER.sample(image_size).to(device)  # (image_size², 2)
        N = tgt_coords.shape[0]

        if isinstance(neural_isometry, EulerianIsometry):
            src_coords = tgt_coords
            src_logabsdet = torch.zeros(N, device=device, dtype=tgt_coords.dtype)
        else:
            with torch.no_grad():
                src_coords, src_logabsdet, _ = neural_isometry.pullback(
                    tgt_coords=tgt_coords,
                    tgt_logabsdet=torch.zeros(N, device=device, dtype=tgt_coords.dtype),
                    tgt_field=torch.zeros(1, N, 1, device=device, dtype=tgt_coords.dtype),
                    **pushforward_kwargs,
                )
            src_coords = src_coords.detach()
            src_logabsdet = src_logabsdet.detach()

        # Fix the atom index set once per epoch (same across accumulation steps).
        indices = initial_dictionary.get_truncated_indices(K).to(device)  # (A, d+1)
        pmfs    = initial_dictionary.get_index_pmfs(indices).to(device)   # (A,)
        A       = indices.shape[0]

        init_atoms = initial_dictionary.get_atoms(src_coords, indices)  # (A, N, C)
        C = init_atoms.shape[-1]

        total_iso_loss      = 0.0
        total_iso_range     = 0.0
        total_iso_concept   = 0.0
        total_mean_loss     = 0.0
        total_mean_concept  = 0.0

        for _ in range(grad_accumulation_steps):

            neural_isometry.shuffle_model_state(**model_state_kwargs)

            # ── Mean function pass ────────────────────────────────────────────
            # Trains mean_function so that mean_img alone looks like the concept.
            # No range penalty — the mean output is unconstrained; only the
            # combined image gets the range penalty in the isometry pass.
            # batch_size copies are passed so the SDS loss draws batch_size
            # independent (t, noise) samples per step, matching the gradient
            # variance of the old single-pass code.
            if mean_function is not None and optim_mean_function is not None:
                mean_img_raw = mean_function(tgt_coords)                                          # (N, C)
                mean_img_raw = mean_img_raw.reshape(image_size, image_size, C).permute(2, 0, 1)  # (C, H, W)
                # tanh as output activation: bounds output to (-1, 1), eliminating the
                # dead zone that clamp+range_penalty created (Adam momentum could push
                # pixels out of [-1,1] and the tiny range penalty gradient couldn't
                # compete with accumulated SDS momentum to pull them back).
                mean_img = mean_img_raw.tanh()                                                    # (C, H, W)
                mean_images = mean_img.unsqueeze(0).expand(batch_size, -1, -1, -1)               # (B, C, H, W)
                mean_concept_val = concept_loss(mean_images, text_embeds)  # already in (-1, 1)
                mean_step = mean_concept_val / grad_accumulation_steps
                mean_step.backward()
                total_mean_loss    += mean_step.item()
                total_mean_concept += mean_concept_val.item() / grad_accumulation_steps
                mean_img_detached = mean_img.detach()
            elif mean_function is not None:
                with torch.no_grad():
                    mean_img_raw = mean_function(tgt_coords).tanh()
                    mean_img_raw = mean_img_raw.reshape(image_size, image_size, C).permute(2, 0, 1)
                mean_img_detached = mean_img_raw
            else:
                mean_img_detached = None

            # ── Isometry pass ─────────────────────────────────────────────────
            # Skipped entirely when variation_strength == 0 so no wasted SDS
            # forward passes fire with zero-gradient tensors.
            if variation_strength > 0.0:
                _, _, pushed_atoms = neural_isometry.pushforward(
                    src_coords=src_coords,
                    src_logabsdet=src_logabsdet,
                    src_field=init_atoms,
                    **pushforward_kwargs,
                )  # (A, N, C)

                coeff_std    = pmfs.pow(coefficient_decay)                            # (A,)
                coefficients = torch.randn(batch_size, A, device=device) * coeff_std # (B, A)
                coefficients = coefficients / torch.norm(coefficients, dim=-1, keepdim=True)

                combo     = torch.einsum("ba,anc->bnc", coefficients, pushed_atoms)
                combo_img = combo.reshape(batch_size, image_size, image_size, C).permute(0, 3, 1, 2)

                images = variation_strength * combo_img
                if mean_img_detached is not None:
                    images = images + mean_img_detached.unsqueeze(0)

                iso_range = (F.relu(-1.0 - images) + F.relu(images - 1.0)).mean()
                iso_concept_val = concept_loss(images.clamp(-1, 1), text_embeds)
                iso_step = (iso_concept_val + range_penalty_weight * iso_range) / grad_accumulation_steps
                iso_step.backward()
                total_iso_loss    += iso_step.item()
                total_iso_range   += iso_range.item()       / grad_accumulation_steps
                total_iso_concept += iso_concept_val.item() / grad_accumulation_steps

        if variation_strength > 0.0 and max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(neural_isometry.parameters(), max_grad_norm)

        if wandb_enabled:
            if mean_function is not None and optim_mean_function is not None:
                wandb.log({"train/mean_total_loss":          total_mean_loss},    step=epoch_i)
                wandb.log({"train/mean_concept_loss":        total_mean_concept}, step=epoch_i)
                wandb.log({"train/grad_norm_mean_function":  get_grad_norm(mean_function)},    step=epoch_i)
                wandb.log({"train/param_norm_mean_function": get_param_norm(mean_function)},   step=epoch_i)
                wandb.log({"train/avg_lr_mean_function":     get_avg_lr(optim_mean_function)}, step=epoch_i)
            if variation_strength > 0.0:
                wandb.log({"train/isometry_total_loss":   total_iso_loss},    step=epoch_i)
                wandb.log({"train/isometry_range_loss":   total_iso_range},   step=epoch_i)
                wandb.log({"train/isometry_concept_loss": total_iso_concept}, step=epoch_i)
                wandb.log({"train/grad_norm_isometry":    get_grad_norm(neural_isometry)},  step=epoch_i)
                wandb.log({"train/param_norm_isometry":   get_param_norm(neural_isometry)}, step=epoch_i)
                wandb.log({"train/avg_lr_isometry":       get_avg_lr(optim_isometry)},      step=epoch_i)

        pbar.set_postfix({
            "iso_loss":  f"{total_iso_loss:.4f}",
            "mean_loss": f"{total_mean_loss:.4f}",
        })

        if mean_function is not None and optim_mean_function is not None:
            optim_mean_function.step()
            optim_mean_function.zero_grad()
            step_scheduler(scheduler_mean_function, total_mean_loss)

        if variation_strength > 0.0:
            optim_isometry.step()
            optim_isometry.zero_grad()
            step_scheduler(scheduler_isometry, total_iso_loss)

        if checkpointer is not None:
            checkpointer.step(optimizer_step=epoch_i + 1, epoch=epoch_i, metric=total_iso_loss)

        for callback in callbacks:
            callback(
                epoch=epoch_i,
                neural_isometry=neural_isometry,
                mean_function=mean_function,
                wandb_enabled=wandb_enabled,
                device=device,
            )


@hydra.main(version_base=None, config_path="conf", config_name="concept_basis")
def main(conf: DictConfig):
    # TODO: if things work out, we don't need the mean_function anymore
    neural_isometry: NeuralIsometry = instantiate(conf.neural_isometry)
    initial_dictionary: InfiDictionary = instantiate(conf.initial_dictionary)
    mean_function: NeuralField | None = (
        instantiate(conf.mean_function) if conf.get("mean_function") is not None else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, wandb_run_id = None, None
    if conf.resume_training.enabled and conf.resume_training.checkpoint_path is not None:
        checkpoint = torch.load(
            conf.resume_training.checkpoint_path, weights_only=False, map_location=device
        )
        run_name = checkpoint.get("run_name", "")
        if run_name and run_name.startswith("wandb-"):
            wandb_run_id = run_name[len("wandb-"):]
    else:
        run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if conf.wandb.enabled:
        wandb_run_name = str(conf.wandb.run_name) if conf.wandb.run_name is not None else None
        tags = (
            [f"{key}:{value}" for key, value in conf.wandb.tags.items()]
            if "tags" in conf.wandb else []
        )
        wandb.init(
            project=conf.wandb.project,
            entity=conf.wandb.entity,
            config=OmegaConf.to_container(conf, resolve=True),
            tags=tags,
            settings=wandb.Settings(start_method="thread"),
            name=wandb_run_name,
            id=wandb_run_id,
            resume="must" if wandb_run_id is not None else None,
        )
        run_name = f"wandb-{wandb.run.id}"
    elif wandb_run_id is not None:
        raise ValueError("Resuming a wandb run without specifying wandb=enabled!")

    if mean_function is not None:
        mean_function = mean_function.to(device)

    isometry_optimizer_callable = instantiate(conf.isometry_optimizer_callable)
    isometry_scheduler_callable = (
        instantiate(conf.isometry_scheduler_callable)
        if conf.get("isometry_scheduler_callable", None)
        else None
    )
    optim_isometry = isometry_optimizer_callable(neural_isometry.parameters())
    sched_isometry = (
        isometry_scheduler_callable(optim_isometry)
        if isometry_scheduler_callable is not None
        else None
    )

    optim_mean_function, sched_mean_function = None, None
    if mean_function is not None:
        mean_function_optimizer_callable = instantiate(conf.mean_function_optimizer_callable)
        mean_function_scheduler_callable = (
            instantiate(conf.mean_function_scheduler_callable)
            if conf.get("mean_function_scheduler_callable", None)
            else None
        )
        optim_mean_function = mean_function_optimizer_callable(mean_function.parameters())
        sched_mean_function = (
            mean_function_scheduler_callable(optim_mean_function)
            if mean_function_scheduler_callable is not None
            else None
        )

    ckpt_cfg = conf.get("checkpointing", {})
    checkpoint_dir = ckpt_cfg.get("checkpoint_dir", None)
    ckpt_models     = {"neural_isometry": neural_isometry}
    ckpt_optimizers = {"isometry": optim_isometry}
    ckpt_schedulers = {"isometry": sched_isometry}
    if mean_function is not None:
        ckpt_models["mean_function"]          = mean_function
        ckpt_optimizers["mean_function"]      = optim_mean_function
        ckpt_schedulers["mean_function"]      = sched_mean_function
    checkpointer = (
        Checkpointer(
            checkpoint_dir=os.path.join(checkpoint_dir, run_name),
            models=ckpt_models,
            optimizers=ckpt_optimizers,
            schedulers=ckpt_schedulers,
            checkpoint_every_n_steps=ckpt_cfg.get("checkpoint_every_n_steps", None),
            checkpoint_window_size=ckpt_cfg.get("checkpoint_window_size", 3),
            higher_is_better=False,
            run_name=run_name,
            config=OmegaConf.to_container(conf, resolve=True),
        )
        if checkpoint_dir is not None
        else None
    )

    callbacks = (
        []
        if "callbacks" not in conf
        else [instantiate(cb) for cb in conf.callbacks.values()]
    )

    # Build the concept loss and encode the text prompt once (reused every step).
    concept_loss: ConceptLoss = instantiate(conf.concept_loss)
    text_embeds = concept_loss.encode_text(conf.caption, device)

    concept_basis_training(
        neural_isometry=neural_isometry,
        initial_dictionary=initial_dictionary,
        concept_loss=concept_loss,
        text_embeds=text_embeds,
        n_epochs=conf.n_epochs,
        device=device,
        optim_isometry=optim_isometry,
        scheduler_isometry=sched_isometry,
        K=conf.K,
        coefficient_decay=conf.coefficient_decay,
        image_size=conf.image_size,
        wandb_enabled=conf.wandb.enabled,
        model_state_kwargs=conf.get("model_state_kwargs", {}) or {},
        pushforward_kwargs=conf.get("pushforward_kwargs", {}) or {},
        checkpointer=checkpointer,
        checkpoint=checkpoint,
        batch_size=conf.batch_size,
        callbacks=callbacks,
        mean_function=mean_function,
        optim_mean_function=optim_mean_function,
        scheduler_mean_function=sched_mean_function,
        range_penalty_weight=conf.get("range_penalty_weight", 1.0),
        max_grad_norm=conf.get("max_grad_norm", None),
        grad_accumulation_steps=conf.get("grad_accumulation_steps", 1),
        variation_strength=conf.get("variation_strength", 1.0),
    )

    if conf.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
