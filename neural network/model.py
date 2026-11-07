"""Modular Deep Neural Network architecture in PyTorch."""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNeuralNetwork(nn.Module):
    """A configurable Deep Neural Network supporting arbitrary hidden layers,

    activations, batch normalization, and dropout.
    """

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

        # Select activation function
        activation_dict = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
        }
        act_cls = activation_dict.get(activation.lower(), nn.ReLU)

        # Build hidden layer sequence
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_cls())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Apply Kaiming (He) normal initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through hidden layers and linear output head."""
        x = self.hidden_layers(x)
        logits = self.output_layer(x)
        return logits

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels (argmax over logits)."""
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class probabilities (softmax over logits)."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def print_summary(self):
        """Display architecture specifications and total parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("=" * 55)
        print("          CUSTOM NEURAL NETWORK SUMMARY")
        print("=" * 55)
        print(f"Input Dimension     : {self.input_dim}")
        print(f"Hidden Dimensions   : {self.hidden_dims}")
        print(f"Output Dimension    : {self.output_dim}")
        print(f"Batch Normalization : {self.use_batch_norm}")
        print(f"Dropout Rate        : {self.dropout_rate}")
        print("-" * 55)
        print(f"Total Parameters    : {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print("=" * 55)


# Alias for unified naming in OMEGA Neural Network Engine
MLP = CustomNeuralNetwork


if __name__ == "__main__":
    net = CustomNeuralNetwork(input_dim=10, hidden_dims=[64, 32], output_dim=3)
    net.print_summary()
    sample_input = torch.randn(4, 10)
    out = net(sample_input)
    print(f"Input shape: {sample_input.shape} -> Output shape: {out.shape}")
