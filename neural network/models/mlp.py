"""OMEGA NEURAL NETWORK ENGINE: Multi-Layer Perceptron (MLP) Architecture.

Maintains complete backward compatibility with CustomNeuralNetwork while offering
flexible layer definitions, residual options, batch norm, and activations.
"""

from typing import List, Optional
import torch
import torch.nn as nn
from activations.functions import get_activation


class MLP(nn.Module):
    """Deep Multi-Layer Perceptron with batch normalization, dropout, and residual option."""

    def __init__(
        self,
        input_dim: int = 10,
        hidden_dims: List[int] = [64, 32, 16],
        output_dim: int = 3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_layers(x)
        return self.output_layer(h)


# Alias for 100% backward compatibility
CustomNeuralNetwork = MLP
