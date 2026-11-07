"""OMEGA NEURAL NETWORK ENGINE: Normalization and Regularization Layers."""

import torch
import torch.nn as nn


class BatchNorm1D(nn.Module):
    """1D Batch Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class BatchNorm2D(nn.Module):
    """2D Spatial Batch Normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(x)


class LayerNorm(nn.Module):
    """Layer Normalization across channel dimension."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Llama/Mistral style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        norm_x = x * torch.rsqrt(variance + self.eps)
        return self.weight * norm_x


class Dropout(nn.Module):
    """Configurable Dropout."""

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x)
