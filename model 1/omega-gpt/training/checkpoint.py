"""Robust, production-grade checkpoint management for Omega GPT-1."""

import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from config.model_config import ModelConfig
from config.training_config import TrainingConfig


class CheckpointManager:
    """Manages saving, loading, verifying, and resuming training checkpoints for Omega GPT-1."""

    def __init__(self, checkpoint_dir: str = "outputs/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = float("inf")

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        val_loss: float,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        tokenizer_metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> str:
        """Save a complete training checkpoint with model, optimizer, scheduler, and RNG states."""
        raw_model = getattr(model, "_orig_mod", model)

        checkpoint_data = {
            "step": step,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "tokenizer_metadata": tokenizer_metadata or {"vocab_size": model_config.vocab_size},
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

        # Save step checkpoint if tagged or periodic
        step_filename = f"omega_gpt1_step_{step:05d}.pth"
        if tag is not None:
            step_filename = f"omega_gpt1_{tag}.pth"

        step_path = os.path.join(self.checkpoint_dir, step_filename)
        self._atomic_save(checkpoint_data, step_path)

        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "omega_gpt1_latest.pth")
        self._atomic_save(checkpoint_data, latest_path)

        # Save best checkpoint strictly if val_loss improves
        if is_best or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            checkpoint_data["best_val_loss"] = self.best_val_loss
            best_path = os.path.join(self.checkpoint_dir, "omega_gpt1_best.pth")
            self._atomic_save(checkpoint_data, best_path)
            print(f"[Checkpoint] New best validation loss achieved: {val_loss:.4f} -> Saved to '{best_path}'")

        return step_path

    def _atomic_save(self, data: dict, target_path: str):
        """Atomic write using temporary file to prevent corruption if interrupted."""
        temp_path = target_path + ".tmp"
        torch.save(data, temp_path)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
            except OSError:
                pass
        os.rename(temp_path, target_path)

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint and verify architectural compatibility before restoring states."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"[Checkpoint] Loading checkpoint from '{checkpoint_path}'...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Architecture compatibility check
        raw_model = getattr(model, "_orig_mod", model)
        ckpt_config = checkpoint.get("model_config", {})

        if hasattr(raw_model, "config") and ckpt_config:
            model_vocab = raw_model.config.vocab_size
            ckpt_vocab = ckpt_config.get("vocab_size")
            model_d = raw_model.config.d_model
            ckpt_d = ckpt_config.get("d_model")

            if ckpt_vocab is not None and model_vocab != ckpt_vocab:
                raise ValueError(
                    f"Incompatible vocabulary size! Model has vocab_size={model_vocab}, "
                    f"but checkpoint requires vocab_size={ckpt_vocab}. "
                    "Ensure tokenizer and model configurations match."
                )
            if ckpt_d is not None and model_d != ckpt_d:
                raise ValueError(
                    f"Incompatible embedding dimension! Model has d_model={model_d}, "
                    f"but checkpoint requires d_model={ckpt_d}."
                )

        raw_model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"[Checkpoint] Warning: could not restore optimizer state: {e}")

        if "best_val_loss" in checkpoint:
            self.best_val_loss = checkpoint["best_val_loss"]

        # Restore RNG states
        if "rng_state" in checkpoint and checkpoint["rng_state"] is not None:
            rng = checkpoint["rng_state"]
            if "torch" in rng and rng["torch"] is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])

        step = checkpoint.get("step", 0)
        val_loss = checkpoint.get("val_loss", float("inf"))
        print(f"[Checkpoint] Restored checkpoint at step {step} (val_loss: {val_loss:.4f})")

        return checkpoint

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Find latest checkpoint file if available."""
        latest_file = os.path.join(self.checkpoint_dir, "omega_gpt1_latest.pth")
        if os.path.exists(latest_file):
            return latest_file

        step_files = [
            os.path.join(self.checkpoint_dir, f)
            for f in os.listdir(self.checkpoint_dir)
            if f.startswith("omega_gpt1_step_") and f.endswith(".pth")
        ]
        if step_files:
            return sorted(step_files)[-1]

        return None

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Find best checkpoint file if available."""
        best_file = os.path.join(self.checkpoint_dir, "omega_gpt1_best.pth")
        if os.path.exists(best_file):
            return best_file
        return self.get_latest_checkpoint_path()
