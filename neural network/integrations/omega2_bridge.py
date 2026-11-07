"""OMEGA NEURAL NETWORK ENGINE: Bridge to OMEGA MODEL 2 (Assistant).

Exposes neural network inference, classification models, and embedding features
for use as tool and brain components in OMEGA Model 2.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from core.tensor_engine import tensor_engine
from models.mlp import MLP


class Omega2Bridge:
    """Interoperability bridge with Model 2 (AI Assistant)."""

    def __init__(self):
        self.device = tensor_engine.get_device()

    def create_classifier_for_assistant(
        self, input_dim: int, num_classes: int, hidden_dims: List[int] = [64, 32]
    ) -> nn.Module:
        """Create a lightweight neural classifier for Model 2's intent or memory routing."""
        return MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            activation="gelu",
            use_batch_norm=False,
            dropout=0.0,
        ).to(self.device)

    def extract_dense_features(self, model: nn.Module, x: torch.Tensor) -> np.ndarray:
        """Extract penultimate feature representations from an MLP model."""
        model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            # Pass through hidden layers excluding the final linear classification projection
            if hasattr(model, "hidden_layers"):
                features = model.hidden_layers(x)
                return features.cpu().numpy()
            return model(x).cpu().numpy()
