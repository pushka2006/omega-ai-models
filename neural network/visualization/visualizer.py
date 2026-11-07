"""OMEGA NEURAL NETWORK ENGINE: Visualizer and Architecture Inspector.

Computes layer tensor dimensions, weight distributions, and gradient flow diagnostics.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class ModelVisualizer:
    """Extracts structural, weight, and gradient telemetry from neural network modules."""

    @staticmethod
    def get_layer_summary(model: nn.Module) -> List[Dict[str, Any]]:
        """Return layer names, types, parameter counts, and shapes."""
        summary = []
        for name, module in model.named_modules():
            if name == "":
                continue
            params = sum(p.numel() for p in module.parameters(recurse=False))
            trainable = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)

            weights = [p for p in module.parameters(recurse=False) if p.dim() >= 2]
            shape = list(weights[0].shape) if weights else []

            summary.append({
                "layer_name": name,
                "layer_type": module.__class__.__name__,
                "params": params,
                "trainable": trainable,
                "tensor_shape": shape,
            })
        return summary

    @staticmethod
    def inspect_weight_distributions(model: nn.Module) -> List[Dict[str, Any]]:
        """Compute mean, std, min, max, and histogram quantiles for all weight tensors."""
        stats = []
        for name, param in model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                data = param.detach().cpu().numpy()
                hist, bin_edges = np.histogram(data, bins=8)
                stats.append({
                    "parameter": name,
                    "shape": list(data.shape),
                    "mean": round(float(np.mean(data)), 5),
                    "std": round(float(np.std(data)), 5),
                    "min": round(float(np.min(data)), 5),
                    "max": round(float(np.max(data)), 5),
                    "histogram": [int(h) for h in hist],
                    "bins": [round(float(b), 4) for b in bin_edges],
                })
        return stats

    @staticmethod
    def inspect_gradient_flow(model: nn.Module) -> Dict[str, Any]:
        """Analyze gradient norms across layers to diagnose vanishing or exploding gradients."""
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.detach().norm(2).item()
                grad_norms[name] = round(norm, 5)

        if not grad_norms:
            return {"status": "No gradients available (Run backward pass first).", "layers": {}}

        max_norm = max(grad_norms.values())
        min_norm = min(grad_norms.values())

        posture = "Stable"
        if max_norm > 10.0:
            posture = "Warning: Exploding gradients detected (consider gradient clipping)."
        elif min_norm < 1e-7:
            posture = "Warning: Vanishing gradients detected in early layers."

        return {
            "status": posture,
            "max_gradient_norm": max_norm,
            "min_gradient_norm": min_norm,
            "layer_gradients": grad_norms,
        }
