"""Evaluation routines for Omega GPT-1: Loss and Perplexity."""

import math
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    max_eval_steps: int = 50,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate model on validation data loader to compute loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_eval_steps:
            break

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            total_steps += 1

    model.train()

    avg_loss = total_loss / max(1, total_steps)
    perplexity = math.exp(min(avg_loss, 20.0))  # Cap to prevent numerical overflow

    return {
        "val_loss": avg_loss,
        "perplexity": perplexity,
        "eval_steps": total_steps,
    }

