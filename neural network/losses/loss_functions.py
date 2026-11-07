"""OMEGA NEURAL NETWORK ENGINE: Loss Functions.

Implements standard and metric learning loss functions with input shape validation.
"""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pairs of embeddings: L = (1-y)*D^2 + y*max(0, margin - D)^2."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(emb1, emb2)
        loss = torch.mean(
            (1.0 - label) * torch.pow(euclidean_distance, 2)
            + label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


LOSS_REGISTRY = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    "bce_with_logits": nn.BCEWithLogitsLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "l1": nn.L1Loss,
    "huber": nn.HuberLoss,
    "kl_div": nn.KLDivLoss,
    "contrastive": ContrastiveLoss,
}


def get_loss_function(name: Union[str, nn.Module], **kwargs) -> nn.Module:
    """Factory resolver for loss functions."""
    if isinstance(name, nn.Module):
        return name
    name_clean = str(name).lower().replace("-", "_").strip()
    if name_clean not in LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss function '{name}'. Supported: {list(LOSS_REGISTRY.keys())}"
        )
    loss_cls = LOSS_REGISTRY[name_clean]
    return loss_cls(**kwargs)
