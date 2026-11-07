"""OMEGA NEURAL NETWORK ENGINE: Convolutional Neural Network (CNN) Architecture."""

from typing import List
import torch
import torch.nn as nn
from activations.functions import get_activation


class ConvNet(nn.Module):
    """Configurable 2D Convolutional Neural Network for spatial and image classification."""

    def __init__(
        self,
        in_channels: int = 1,
        channel_dims: List[int] = [32, 64, 128],
        num_classes: int = 10,
        activation: str = "relu",
        dropout: float = 0.2,
    ):
        super().__init__()
        conv_layers = []
        prev_c = in_channels

        for c_dim in channel_dims:
            conv_layers.extend([
                nn.Conv2d(prev_c, c_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_dim),
                get_activation(activation),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(dropout),
            ])
            prev_c = c_dim

        self.features = nn.Sequential(*conv_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(prev_c * 4, 128),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        pooled = self.avgpool(feat)
        return self.classifier(pooled)
