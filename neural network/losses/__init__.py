"""OMEGA NEURAL NETWORK ENGINE Losses Package."""

from .loss_functions import LOSS_REGISTRY, ContrastiveLoss, get_loss_function

__all__ = ["LOSS_REGISTRY", "ContrastiveLoss", "get_loss_function"]
