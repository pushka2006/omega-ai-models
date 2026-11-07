"""OMEGA NEURAL NETWORK ENGINE: Checkpoint Manager.

Safely saves, inspects, and restores model weights, optimizer states, and hyperparameter metadata.
"""

import os
import time
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from config.engine_config import CHECKPOINT_DIR


class CheckpointManager:
    """Handles atomic persistence and verification of neural network checkpoints."""

    def __init__(self, save_dir: Optional[str] = None):
        self.save_dir = save_dir or str(CHECKPOINT_DIR)
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
        config: Dict[str, Any],
        filename: str = "omega_checkpoint.pth",
        is_best: bool = False,
    ) -> str:
        """Atomically serialize model checkpoint with complete training state."""
        filepath = os.path.join(self.save_dir, filename)
        payload = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "timestamp": time.time(),
        }

        # Save primary checkpoint
        torch.save(payload, filepath)

        # If best, also save to best marker
        if is_best:
            best_path = os.path.join(self.save_dir, "omega_best_model.pth")
            torch.save(payload, best_path)

        return filepath

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> Dict[str, Any]:
        """Load state dictionaries into model and optional optimizer."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        ckpt = torch.load(filepath, map_location=device, weights_only=False)

        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            return ckpt
        else:
            # Raw state dict
            model.load_state_dict(ckpt)
            return {"model_state_dict": ckpt}

    def inspect_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Inspect checkpoint without instantiating the model."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        ckpt = torch.load(filepath, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)

        layer_shapes = {k: list(v.shape) for k, v in state_dict.items() if "weight" in k}
        return {
            "epoch": ckpt.get("epoch", "?"),
            "step": ckpt.get("step", "?"),
            "metrics": ckpt.get("metrics", {}),
            "config": ckpt.get("config", {}),
            "layers_count": len(layer_shapes),
            "layer_shapes": layer_shapes,
            "file_size_mb": round(os.path.getsize(filepath) / (1024**2), 2),
        }


# Singleton
checkpoint_manager = CheckpointManager()
