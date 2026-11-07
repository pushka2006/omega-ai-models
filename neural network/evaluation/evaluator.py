"""OMEGA NEURAL NETWORK ENGINE: Model Evaluator.

Computes accuracy, precision, recall, F1-score, and regression metrics (MSE, MAE).
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Evaluator:
    """Evaluates neural network predictions over validation and test loaders."""

    @staticmethod
    @torch.no_grad()
    def evaluate_classifier(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute multi-class validation loss, accuracy, and macro metrics."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0

        all_preds = []
        all_targets = []

        for x_b, y_b in data_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            logits = model(x_b)
            loss = criterion(logits, y_b)

            total_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=-1)

            correct += (preds == y_b).sum().item()
            total_samples += len(y_b)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_b.cpu().numpy())

        avg_loss = total_loss / max(1, total_samples)
        accuracy = (correct / max(1, total_samples)) * 100.0

        # Compute Macro Precision, Recall, F1 in NumPy
        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        classes = np.unique(targets_arr)

        precisions, recalls, f1s = [], [], []
        for c in classes:
            tp = np.sum((preds_arr == c) & (targets_arr == c))
            fp = np.sum((preds_arr == c) & (targets_arr != c))
            fn = np.sum((preds_arr != c) & (targets_arr == c))

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        return {
            "loss": round(avg_loss, 4),
            "accuracy": round(accuracy, 2),
            "macro_precision": round(float(np.mean(precisions)), 4) if precisions else 0.0,
            "macro_recall": round(float(np.mean(recalls)), 4) if recalls else 0.0,
            "macro_f1": round(float(np.mean(f1s)), 4) if f1s else 0.0,
            "total_samples": total_samples,
        }

    @staticmethod
    @torch.no_grad()
    def evaluate_regressor(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """Compute regression MSE and MAE metrics."""
        model.eval()
        total_loss = 0.0
        total_samples = 0
        all_abs_errors = []

        for x_b, y_b in data_loader:
            x_b = x_b.to(device)
            y_b = y_b.to(device)

            preds = model(x_b)
            loss = criterion(preds, y_b)

            total_loss += loss.item() * len(y_b)
            total_samples += len(y_b)
            all_abs_errors.extend(torch.abs(preds - y_b).cpu().numpy().flatten())

        avg_loss = total_loss / max(1, total_samples)
        mae = float(np.mean(all_abs_errors)) if all_abs_errors else 0.0

        return {
            "loss": round(avg_loss, 4),
            "mae": round(mae, 4),
            "total_samples": total_samples,
        }
