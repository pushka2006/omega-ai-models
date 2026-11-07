"""Learning rate scheduler with linear warmup and cosine decay."""

import math
import torch
from config.training_config import TrainingConfig


class CosineWarmupScheduler:
    """Cosine learning rate scheduler with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, config: TrainingConfig):
        self.optimizer = optimizer
        self.learning_rate = config.learning_rate
        self.min_learning_rate = config.min_learning_rate
        self.warmup_steps = config.warmup_steps
        self.decay_steps = config.decay_steps

    def get_lr(self, step: int) -> float:
        """Calculate learning rate at step."""
        # 1. Linear warmup
        if step < self.warmup_steps:
            return self.learning_rate * (step + 1) / max(1, self.warmup_steps)

        # 2. Beyond decay steps, keep minimum lr
        if step > self.decay_steps:
            return self.min_learning_rate

        # 3. Cosine decay phase
        decay_ratio = (step - self.warmup_steps) / max(1, (self.decay_steps - self.warmup_steps))
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_learning_rate + coeff * (self.learning_rate - self.min_learning_rate)

    def step(self, step: int) -> float:
        """Update optimizer learning rate and return new value."""
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

