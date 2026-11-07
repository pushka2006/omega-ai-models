"""OMEGA NEURAL NETWORK ENGINE: Declarative Model Builder.

Builds and validates arbitrary neural networks from YAML, JSON, or Python dictionary specifications.
"""

import json
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from activations.functions import get_activation
from layers.dense import Dense
from layers.normalization import BatchNorm1D, Dropout, LayerNorm


class ConfiguredSequentialModel(nn.Module):
    """Dynamically built model conforming to a declarative architecture specification."""

    def __init__(self, name: str, layers: List[nn.Module], input_size: int, output_size: int, config: Dict[str, Any]):
        super().__init__()
        self.model_name = name
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelBuilder:
    """Parses, validates, and instantiates neural architectures from declarative configs."""

    @staticmethod
    def build_from_config(config: Union[Dict[str, Any], str]) -> ConfiguredSequentialModel:
        """Construct a neural network module from dictionary or JSON string."""
        if isinstance(config, str):
            config = json.loads(config)

        model_spec = config.get("model", config)
        name = model_spec.get("name", "omega_custom_model")
        input_size = model_spec.get("input_size")
        if input_size is None:
            raise ValueError("Model specification must include 'input_size'.")

        layers_spec = model_spec.get("layers", [])
        if not layers_spec:
            raise ValueError("Model specification must contain at least one layer in 'layers'.")

        built_layers = []
        prev_dim = int(input_size)

        for i, l_cfg in enumerate(layers_spec):
            l_type = l_cfg.get("type", "").lower().strip()

            if l_type in ["linear", "dense"]:
                units = l_cfg.get("units") or l_cfg.get("out_features")
                if not units:
                    raise ValueError(f"Layer {i} (linear/dense) must specify 'units'.")
                built_layers.append(Dense(prev_dim, int(units)))
                prev_dim = int(units)

                act = l_cfg.get("activation")
                if act:
                    built_layers.append(get_activation(act))

                if l_cfg.get("batch_norm", False):
                    built_layers.append(BatchNorm1D(prev_dim))

            elif l_type == "dropout":
                rate = float(l_cfg.get("rate", l_cfg.get("p", 0.1)))
                built_layers.append(Dropout(rate))

            elif l_type == "batch_norm":
                built_layers.append(BatchNorm1D(prev_dim))

            elif l_type == "layer_norm":
                built_layers.append(LayerNorm(prev_dim))

            elif l_type == "activation":
                act_name = l_cfg.get("name", "relu")
                built_layers.append(get_activation(act_name))

            else:
                raise ValueError(f"Unsupported layer type '{l_type}' at layer index {i}.")

        return ConfiguredSequentialModel(
            name=name,
            layers=built_layers,
            input_size=input_size,
            output_size=prev_dim,
            config=model_spec,
        )


# Singleton
model_builder = ModelBuilder()
