"""OMEGA NEURAL NETWORK ENGINE: Production Training Engine.

Supports forward/backward execution, validation, early stopping,
gradient clipping, learning rate scheduling, mixed precision (AMP), and checkpoint persistence.
"""

import time
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from checkpoints.checkpoint_manager import checkpoint_manager
from core.tensor_engine import tensor_engine
from evaluation.evaluator import Evaluator
from optimizers.optimizer_factory import clip_gradients


class TrainingEngine:
    """Configurable neural network trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[torch.device] = None,
        scheduler: Optional[Any] = None,
        grad_clip: float = 1.0,
        use_amp: bool = False,
    ):
        self.device = device or tensor_engine.get_device()
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None

        self.history: List[Dict[str, Any]] = []

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Execute one full training epoch over the dataset."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        t0 = time.time()

        for x_b, y_b in train_loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self.scaler:
                with torch.cuda.amp.autocast():
                    preds = self.model(x_b)
                    loss = self.criterion(preds, y_b)
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    clip_gradients(self.model, self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                preds = self.model(x_b)
                loss = self.criterion(preds, y_b)
                loss.backward()
                if self.grad_clip > 0:
                    clip_gradients(self.model, self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item() * len(y_b)
            total_samples += len(y_b)

        elapsed = time.time() - t0
        throughput = total_samples / elapsed if elapsed > 0 else 0
        return {
            "loss": round(total_loss / max(1, total_samples), 4),
            "samples_per_sec": round(throughput, 1),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        checkpoint_name: str = "omega_model.pth",
        task_type: str = "classification",  # 'classification' or 'regression'
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping, validation, and checkpointing."""
        best_val_metric = float("inf") if task_type == "regression" else -1.0
        patience_counter = 0
        total_t0 = time.time()

        print("=" * 65)
        print("          OMEGA NEURAL NETWORK TRAINING EXECUTION")
        print("=" * 65)
        print(f"Device: {self.device} | Epochs: {epochs} | Task: {task_type.upper()}")

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)

            # Evaluate on validation split
            if task_type == "classification":
                val_metrics = Evaluator.evaluate_classifier(
                    self.model, val_loader, self.criterion, self.device
                )
                current_metric = val_metrics["accuracy"]
                is_best = current_metric > best_val_metric
            else:
                val_metrics = Evaluator.evaluate_regressor(
                    self.model, val_loader, self.criterion, self.device
                )
                current_metric = val_metrics["loss"]
                is_best = current_metric < best_val_metric

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            # Early stopping & checkpoint tracking
            if is_best:
                best_val_metric = current_metric
                patience_counter = 0
                checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=epoch * len(train_loader),
                    metrics=val_metrics,
                    config={"epochs": epochs, "task": task_type},
                    filename=checkpoint_name,
                    is_best=True,
                )
                star = " ★ [BEST]"
            else:
                patience_counter += 1
                star = ""

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_metric": current_metric,
                "lr": current_lr,
                "samples_per_sec": train_metrics["samples_per_sec"],
                "is_best": is_best,
            }
            if task_type == "classification":
                record["val_acc"] = val_metrics["accuracy"]
                record["macro_f1"] = val_metrics["macro_f1"]

            self.history.append(record)

            if on_epoch_end:
                on_epoch_end(record)

            metric_str = f"Val Acc: {val_metrics['accuracy']:.2f}%" if task_type == "classification" else f"Val MAE: {val_metrics['mae']:.4f}"
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"{metric_str} | "
                f"LR: {current_lr:.2e} | "
                f"{train_metrics['samples_per_sec']:.0f} samp/s{star}"
            )

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (Patience: {early_stopping_patience}).")
                break

        total_time = round(time.time() - total_t0, 2)
        print("-" * 65)
        print(f"Training completed in {total_time}s | Best Val Metric: {best_val_metric:.4f}")
        return {
            "history": self.history,
            "best_metric": best_val_metric,
            "epochs_trained": len(self.history),
            "training_time_seconds": total_time,
        }
