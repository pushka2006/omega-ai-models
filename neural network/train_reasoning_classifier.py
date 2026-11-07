"""OMEGA NEURAL NETWORK ENGINE: Train Deep Reasoning Complexity Model on Fable Dataset.

Uses the engine's modular layers, AdamW optimizer, Cosine Annealing,
and SQLite experiment tracking.
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from optimizers.optimizer_factory import configure_optimizers, LinearWarmupCosineDecayScheduler
from losses.loss_functions import get_loss_function
from evaluation.evaluator import Evaluator
from experiments.tracker import ExperimentTracker

# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets_hub" / "processed" / "fable_reasoning_neural.npz"
CKPT_DIR = BASE_DIR / "checkpoints_saved"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = CKPT_DIR / "fable_reasoning_complexity_model.pth"


def train_reasoning_model(epochs: int = 35, batch_size: int = 32, lr: float = 0.008):
    print("=" * 65)
    print("  OMEGA NEURAL NETWORK: FABLE REASONING COMPLEXITY CLASSIFIER")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at '{DATA_PATH}'. Run data ingestion first.")

    data = np.load(DATA_PATH)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)
    print(f"Loaded {len(X)} samples with {X.size(1)} features across {len(torch.unique(y))} classes.")

    # Train / Val Split (80% / 20%)
    torch.manual_seed(42)
    indices = torch.randperm(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Initialize Engine MLP
    model = MLP(
        input_dim=8,
        hidden_dims=[64, 32, 16],
        output_dim=4,
        activation="gelu",
        use_batch_norm=True,
        dropout=0.1,
    )

    criterion = get_loss_function("cross_entropy")
    optimizer = configure_optimizers(model, optimizer_name="adamw", learning_rate=lr, weight_decay=0.01)
    scheduler = LinearWarmupCosineDecayScheduler(optimizer, warmup_steps=5, total_steps=epochs, min_lr=1e-4)

    tracker = ExperimentTracker()
    start_time = time.time()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(yb)
            preds = torch.argmax(logits, dim=-1)
            train_correct += (preds == yb).sum().item()
            total_train += len(yb)

        train_loss /= total_train
        train_acc = (train_correct / total_train) * 100.0

        # Evaluate validation
        metrics = Evaluator.evaluate_classifier(model, val_loader, criterion, torch.device("cpu"))
        val_loss = metrics["loss"]
        val_acc = metrics["accuracy"]
        macro_f1 = metrics.get("macro_f1", 0.0)

        scheduler.step()
        curr_lr = scheduler.get_last_lr()[0]

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": 8,
                "hidden_dims": [64, 32, 16],
                "output_dim": 4,
                "val_accuracy": val_acc,
                "macro_f1": macro_f1,
                "epoch": epoch,
            }, SAVE_PATH)

        star = " ★ [NEW BEST]" if is_best else ""
        if epoch % 5 == 0 or epoch == epochs or is_best:
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | "
                f"Macro F1: {macro_f1:.1f}% | LR: {curr_lr:.2e}{star}"
            )

    duration = time.time() - start_time
    tracker.log_run(
        name="fable_reasoning_complexity",
        model_config={"arch": "MLP", "input_dim": 8, "hidden_dims": [64, 32, 16], "output_dim": 4},
        dataset_info={"name": "Fable-5.1-Max-Reasoning", "samples": len(X)},
        hyperparameters={"epochs": epochs, "batch_size": batch_size, "lr": lr, "optimizer": "adamw"},
        best_val_metric=best_val_acc,
        final_train_loss=train_loss,
        epochs_trained=epochs,
        duration_seconds=round(duration, 2),
        checkpoint_path=str(SAVE_PATH),
    )
    print("-" * 65)
    print(f"Training Complete! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved successfully to: '{SAVE_PATH}'")
    print("=" * 65)
    return best_val_acc


if __name__ == "__main__":
    train_reasoning_model()
