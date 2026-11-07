"""OMEGA NEURAL NETWORK ENGINE: Activation Functions.

Provides configurable activation implementations and factory dispatch.
"""

from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """Swish / SiLU activation function: x * sigmoid(beta * x)."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)


ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": Swish,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": lambda **kw: nn.Softmax(dim=kw.get("dim", -1)),
    "elu": nn.ELU,
}


def get_activation(name: Union[str, nn.Module], **kwargs) -> nn.Module:
    """Factory resolver for activation functions."""
    if isinstance(name, nn.Module):
        return name
    name_clean = str(name).lower().replace("-", "_").strip()
    if name_clean not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation '{name}'. Supported: {list(ACTIVATION_REGISTRY.keys())}"
        )
    act_cls = ACTIVATION_REGISTRY[name_clean]
    return act_cls(**kwargs) if callable(act_cls) else act_cls()
