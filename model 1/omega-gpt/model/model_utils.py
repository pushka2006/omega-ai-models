"""Model utilities for Omega GPT-1: parameter accounting and weight initialization."""

import math
from typing import Tuple
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_parameter_count(num_params: int) -> str:
    """Format parameter count into human-readable string (e.g., 3.12M, 500K)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def init_weights(module: nn.Module, initializer_range: float = 0.02, num_layers: int = 4):
    """Custom weight initialization following GPT-2 standard.

    Scales residual projection layers by 1 / sqrt(2 * num_layers).
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)

    # Special scaled initialization for residual output projections
    for name, p in module.named_parameters():
        if name.endswith("c_proj.weight"):
            nn.init.normal_(p, mean=0.0, std=initializer_range / math.sqrt(2 * num_layers))

