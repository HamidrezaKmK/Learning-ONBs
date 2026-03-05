import math
import os
from collections import deque

import torch
import torch.nn as nn
import yaml


class Checkpointer:
    """
    Generic checkpoint manager for any combination of models, optimizers, and schedulers.

    Components are passed as named dicts so the class is not tied to any specific
    architecture. The checkpoint format stores state dicts nested under "models",
    "optimizers", and "schedulers" keys.

    Usage::

        checkpointer = Checkpointer(
            checkpoint_dir="checkpoints",
            models={"encoder": encoder, "decoder": decoder},
            optimizers={"main": optimizer},
            schedulers={"main": scheduler},
            checkpoint_every_n_steps=500,
            checkpoint_window_size=3,
            run_name=run_name,
        )

        # Restore from a pre-loaded checkpoint dict (e.g. from torch.load):
        start_epoch, best_metric = checkpointer.restore(checkpoint)

        # During training, call once per optimizer step:
        checkpointer.step(optimizer_step=step, epoch=epoch_i, metric=energy)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        models: dict[str, nn.Module],
        optimizers: dict[str, torch.optim.Optimizer],
        schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler | None],
        checkpoint_every_n_steps: int | None = None,
        checkpoint_window_size: int = 3,
        run_name: str | None = None,
        higher_is_better: bool = True,
        config: dict | None = None,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.models = models
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.checkpoint_every_n_steps = checkpoint_every_n_steps
        self.checkpoint_window_size = checkpoint_window_size
        self.run_name = run_name
        self.higher_is_better = higher_is_better

        self.best_metric = -math.inf if higher_is_better else math.inf
        self.periodic_checkpoints: deque = deque()

        if config is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def restore(self, checkpoint: dict) -> tuple[int, float]:
        """
        Restore all component state dicts from a checkpoint dict.

        Returns:
            (start_epoch, best_metric) — use start_epoch as the first epoch of the
            resumed loop and best_metric to seed the best-checkpoint comparison.
        """
        for name, model in self.models.items():
            model.load_state_dict(checkpoint["models"][name])
        for name, optim in self.optimizers.items():
            optim.load_state_dict(checkpoint["optimizers"][name])
        for name, sched in self.schedulers.items():
            saved = checkpoint["schedulers"].get(name)
            if sched is not None and saved is not None:
                sched.load_state_dict(saved)

        start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        return start_epoch, self.best_metric

    def step(self, optimizer_step: int, epoch: int, metric: float) -> None:
        """
        Evaluate metric and save checkpoints as appropriate:
        - Saves ``best.pt`` whenever the metric improves.
        - Saves ``step_N.pt`` every ``checkpoint_every_n_steps`` optimizer steps,
          evicting the oldest when the window is full.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt = self._build_checkpoint(epoch, metric)

        improved = (
            metric > self.best_metric if self.higher_is_better
            else metric < self.best_metric
        )
        if improved:
            self.best_metric = metric
            ckpt["best_metric"] = self.best_metric
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best.pt"))

        if self.checkpoint_every_n_steps is not None and optimizer_step % self.checkpoint_every_n_steps == 0:
            if len(self.periodic_checkpoints) >= self.checkpoint_window_size:
                old_path = self.periodic_checkpoints.popleft()
                if os.path.exists(old_path):
                    os.remove(old_path)
            periodic_path = os.path.join(self.checkpoint_dir, f"step_{optimizer_step}.pt")
            torch.save(ckpt, periodic_path)
            self.periodic_checkpoints.append(periodic_path)

    def _build_checkpoint(self, epoch: int, metric: float) -> dict:
        return {
            "epoch": epoch,
            "metric": metric,
            "best_metric": self.best_metric,
            "run_name": self.run_name,
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optimizers": {name: optim.state_dict() for name, optim in self.optimizers.items()},
            "schedulers": {
                name: sched.state_dict() if sched is not None else None
                for name, sched in self.schedulers.items()
            },
        }
