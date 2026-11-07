"""OMEGA NEURAL NETWORK ENGINE: Dense / Linear Layer."""

from typing import Optional
import torch
import torch.nn as nn


class Dense(nn.Module):
    """Linear / Dense projection layer with configurable initialization and bias."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_type: str = "kaiming",  # 'kaiming', 'xavier', 'normal', 'zeros'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights(init_type)

    def _init_weights(self, init_type: str):
        t = init_type.lower()
        if t == "kaiming":
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        elif t == "xavier":
            nn.init.xavier_uniform_(self.linear.weight)
        elif t == "normal":
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        elif t == "zeros":
            nn.init.zeros_(self.linear.weight)

        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
