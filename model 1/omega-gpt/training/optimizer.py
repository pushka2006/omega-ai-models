"""Optimizer builder for Omega GPT-1."""

import torch
from config.training_config import TrainingConfig
from model.gpt import OmegaGPT


def build_optimizer(
    model: OmegaGPT, config: TrainingConfig, device_type: str = "cpu"
) -> torch.optim.Optimizer:
    """Builds AdamW optimizer with selective weight decay."""
    return model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device_type,
    )

