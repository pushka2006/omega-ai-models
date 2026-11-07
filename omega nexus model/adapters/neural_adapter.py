"""OMEGA NEXUS MODEL: Adapter for OMEGA NEURAL NETWORK ENGINE (Model 3).

Dynamically loads Model 3 without sys.modules collisions.
"""

from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional
import torch

from config import MODEL_3_DIR


class NeuralAdapter:
    """Interoperability wrapper around OMEGA Neural Network Engine."""

    def __init__(self):
        self.available = False
        self.model = None
        self._load()

    def _load(self):
        try:
            model_file = MODEL_3_DIR / "model.py"
            if not model_file.exists():
                return

            spec_model = importlib.util.spec_from_file_location("omega_neural_model", str(model_file))
            mod_model = importlib.util.module_from_spec(spec_model)
            spec_model.loader.exec_module(mod_model)

            # Load checkpoint if available
            ckpt_file = MODEL_3_DIR / "neural_network_model.pth"
            if ckpt_file.exists():
                ckpt = torch.load(str(ckpt_file), map_location="cpu")
                self.model = mod_model.CustomNeuralNetwork(
                    input_dim=ckpt.get("input_dim", 2),
                    hidden_dims=ckpt.get("hidden_dims", [64, 32, 16]),
                    output_dim=ckpt.get("output_dim", 3),
                )
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.model.eval()
                self.available = True
            else:
                self.model = mod_model.CustomNeuralNetwork(input_dim=2, hidden_dims=[32, 16], output_dim=3)
                self.model.eval()
                self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        """Classify input feature tensor."""
        if not self.available or self.model is None:
            return torch.zeros(features.size(0), dtype=torch.long)
        with torch.no_grad():
            return self.model.predict(features)

    def extract_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract continuous penultimate hidden activations."""
        if not self.available or self.model is None:
            return features.repeat(1, 16)[:, :32]

        with torch.no_grad():
            x = features
            for layer in self.model.hidden_layers:
                x = layer(x)
            return x
