"""Evaluation, visualization, and unit tests for Custom Neural Network."""

import os
import unittest
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
from model import CustomNeuralNetwork
from train import generate_synthetic_data


def evaluate_saved_model(checkpoint_path: str = "neural_network_model.pth"):
    """Load trained model checkpoint and evaluate accuracy and decision boundaries."""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Model checkpoint not found at '{checkpoint_path}'. Run 'python train.py' first.")
        return

    print(f"[Evaluation] Loading model from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model = CustomNeuralNetwork(
        input_dim=checkpoint["input_dim"],
        hidden_dims=checkpoint["hidden_dims"],
        output_dim=checkpoint["output_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate held-out test data
    X_test, y_test = generate_synthetic_data(num_samples=600, num_classes=3, noise=0.25)

    with torch.no_grad():
        preds = model.predict(X_test)
        correct = (preds == y_test).sum().item()
        accuracy = (correct / len(y_test)) * 100.0

    print("=" * 50)
    print("      NEURAL NETWORK EVALUATION REPORT")
    print("=" * 50)
    print(f"Test Samples        : {len(X_test):,}")
    print(f"Test Accuracy       : {accuracy:.2f}%")
    for c in range(3):
        mask = y_test == c
        c_acc = (preds[mask] == c).sum().item() / max(1, mask.sum().item()) * 100.0
        print(f"  Class {c} Accuracy : {c_acc:.1f}%")
    print("=" * 50)

    # Plot Decision Boundary
    plot_decision_boundary(model, X_test, y_test)


def plot_decision_boundary(model, X, y, output_file: str = "decision_boundary.png"):
    """Visualize 2D classification decision boundary formed by the neural network."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    with torch.no_grad():
        Z = model.predict(grid)
        Z = Z.reshape(xx.shape).numpy()

    plt.figure(figsize=(8, 7))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Spectral, edgecolors="k", linewidth=0.5)
    plt.title("Custom Neural Network - Non-Linear Decision Boundary", fontsize=14, fontweight="bold")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"[Evaluation] Decision boundary plot saved to '{output_file}'")


# Automated Unit Tests
class TestCustomNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.model = CustomNeuralNetwork(input_dim=8, hidden_dims=[32, 16], output_dim=4)

    def test_forward_shape(self):
        x = torch.randn(10, 8)
        out = self.model(x)
        self.assertEqual(out.shape, (10, 4))

    def test_prediction_methods(self):
        x = torch.randn(5, 8)
        preds = self.model.predict(x)
        probs = self.model.predict_proba(x)

        self.assertEqual(preds.shape, (5,))
        self.assertEqual(probs.shape, (5, 4))
        # Sum of probabilities should be ~1
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(5), atol=1e-5))

    def test_backprop_gradients(self):
        x = torch.randn(4, 8)
        y = torch.tensor([0, 1, 2, 3])
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)

        self.model.train()
        loss = criterion(self.model(x), y)
        loss.backward()

        has_grad = any(p.grad is not None and torch.norm(p.grad) > 0 for p in self.model.parameters())
        self.assertTrue(has_grad)
        optimizer.step()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=["first-arg-is-ignored"])
    else:
        evaluate_saved_model()
