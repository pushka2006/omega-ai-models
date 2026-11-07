"""OMEGA NEURAL NETWORK ENGINE: Pooling Layers."""

from typing import Tuple, Union
import torch
import torch.nn as nn


class MaxPool1D(nn.Module):
    """1D Max Pooling."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class MaxPool2D(nn.Module):
    """2D Max Pooling."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AvgPool2D(nn.Module):
    """2D Average Pooling."""

    def __init__(self, kernel_size: int = 2, stride: int = 2):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)


class AdaptiveAvgPool2D(nn.Module):
    """Adaptive 2D Average Pooling for scale-invariant feature maps."""

    def __init__(self, output_size: Union[int, Tuple[int, int]] = (1, 1)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)
