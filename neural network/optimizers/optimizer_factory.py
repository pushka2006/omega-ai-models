"""OMEGA NEURAL NETWORK ENGINE: Optimization Engine.

Provides optimizer factory, decoupled weight decay parameter grouping,
learning rate schedulers, and gradient clipping utilities.
"""

import math
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR, _LRScheduler


class LinearWarmupCosineDecayScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing learning rate schedule."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            return [base_lr * (step + 1) / self.warmup_steps for base_lr in self.base_lrs]
        if step > self.total_steps:
            return [self.min_lr for _ in self.base_lrs]

        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.min_lr + coeff * (base_lr - self.min_lr) for base_lr in self.base_lrs]


OPTIMIZER_REGISTRY = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
}


def configure_optimizers(
    model: nn.Module,
    optimizer_name: str = "adamw",
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> torch.optim.Optimizer:
    """Configure optimizer with weight decay applied to 2D weights and excluded from 1D biases/norms."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    opt_key = optimizer_name.lower().strip()
    if opt_key == "adamw":
        return AdamW(groups, lr=learning_rate, betas=betas)
    elif opt_key == "adam":
        return Adam(groups, lr=learning_rate, betas=betas)
    elif opt_key == "sgd":
        return SGD(groups, lr=learning_rate, momentum=momentum)
    elif opt_key == "rmsprop":
        return RMSprop(groups, lr=learning_rate, momentum=momentum)
    elif opt_key == "adagrad":
        return Adagrad(groups, lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Supported: {list(OPTIMIZER_REGISTRY.keys())}")


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    """Clips gradient norm of model parameters."""
    if max_norm > 0:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()
    return 0.0
