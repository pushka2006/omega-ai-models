"""Training script for Custom Neural Network in PyTorch."""

import argparse
import os
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import CustomNeuralNetwork


def generate_synthetic_data(num_samples: int = 1500, num_classes: int = 3, noise: float = 0.2):
    """Generate a non-linear 2D classification dataset (interlocking spirals)."""
    np.random.seed(42)
    samples_per_class = num_samples // num_classes
    X = np.zeros((num_samples, 2), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    for c in range(num_classes):
        ix = range(samples_per_class * c, samples_per_class * (c + 1))
        r = np.linspace(0.1, 1.0, samples_per_class)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, samples_per_class) + np.random.randn(samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c

    return torch.tensor(X), torch.tensor(y)


def train_model(
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    hidden_dims: list = [64, 32, 16],
    save_path: str = "neural_network_model.pth",
):
    """Train CustomNeuralNetwork on synthetic dataset and save metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Using compute device: {device}")

    # Generate data (2D input for easy visualization, 3 classes)
    X, y = generate_synthetic_data(num_samples=1800, num_classes=3)

    # Shuffle data before train/val split
    indices = torch.randperm(len(X))
    X, y = X[indices], y[indices]

    # Train / Val split (80% / 20%)
    split = int(0.8 * len(X))
    train_ds = TensorDataset(X[:split], y[:split])
    val_ds = TensorDataset(X[split:], y[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer, Scheduler
    model = CustomNeuralNetwork(
        input_dim=2,
        hidden_dims=hidden_dims,
        output_dim=3,
        activation="relu",
        use_batch_norm=True,
        dropout=0.1,
    ).to(device)

    model.print_summary()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0

    print("=" * 60)
    print(f"TRAINING FOR {epochs} EPOCHS (Batch Size: {batch_size}, LR: {learning_rate})")
    print("=" * 60)

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            preds = torch.argmax(outputs, dim=-1)
            correct_train += (preds == batch_y).sum().item()
            total_train += len(batch_y)

        scheduler.step()

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100.0

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * len(batch_y)
                preds = torch.argmax(outputs, dim=-1)
                correct_val += (preds == batch_y).sum().item()
                total_val += len(batch_y)

        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "hidden_dims": hidden_dims,
                    "input_dim": 2,
                    "output_dim": 3,
                },
                save_path,
            )

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:3d}/{epochs:3d} | "
                f"Train Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.1f}% | "
                f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.1f}%"
            )

    print("=" * 60)
    print(f"TRAINING COMPLETED! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model checkpoint saved to '{save_path}'")
    print("=" * 60)

    # Save training curves plot
    save_training_plots(history)


def save_training_plots(history: dict, output_file: str = "training_metrics.png"):
    """Plot and save loss and accuracy progression."""
    if plt is None:
        print("[Training] Notice: matplotlib is not installed. Skipping plot generation.")
        return
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="crimson")
    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", color="darkgreen")
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="darkorange")
    ax2.set_title("Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    print(f"[Training] Training curve saved to '{output_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom Neural Network")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
