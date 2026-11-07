"""OMEGA NEURAL NETWORK ENGINE: Dataset Engine.

Loads structured tabular, text, and synthetic scientific datasets.
Generates balanced multi-class spiral, moons, and high-dimensional regression benchmarks.
"""

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class OmegaDataset(Dataset):
    """General PyTorch Dataset wrapper over feature and target tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        assert len(x) == len(y), f"Features length ({len(x)}) does not match targets length ({len(y)})"
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class SyntheticDatasetGenerator:
    """Generates synthetic benchmark datasets for non-linear classification and regression."""

    @staticmethod
    def generate_spiral(
        samples_per_class: int = 400,
        num_classes: int = 3,
        noise: float = 0.2,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate classic 2D non-linear Archimedean spiral classification problem."""
        np.random.seed(seed)
        total_samples = samples_per_class * num_classes
        x = np.zeros((total_samples, 2), dtype=np.float32)
        y = np.zeros(total_samples, dtype=np.int64)

        for j in range(num_classes):
            ix = range(samples_per_class * j, samples_per_class * (j + 1))
            r = np.linspace(0.0, 1.0, samples_per_class)
            t = np.linspace(j * 4, (j + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * noise
            x[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            y[ix] = j

        # Shuffle indices to prevent class-split leakage
        perm = np.random.permutation(total_samples)
        x_shuffled = torch.tensor(x[perm], dtype=torch.float32)
        y_shuffled = torch.tensor(y[perm], dtype=torch.long)
        return x_shuffled, y_shuffled

    @staticmethod
    def generate_linear_regression(
        num_samples: int = 1000,
        in_features: int = 10,
        noise: float = 0.1,
        seed: int = 42,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate high-dimensional multi-feature regression dataset."""
        np.random.seed(seed)
        true_w = np.random.randn(in_features, 1)
        x = np.random.randn(num_samples, in_features).astype(np.float32)
        y = (x @ true_w + np.random.randn(num_samples, 1) * noise).astype(np.float32)
        return torch.tensor(x), torch.tensor(y)


class DatasetLoader:
    """Loads CSV, JSON, and synthetic datasets into training-ready DataLoaders."""

    @staticmethod
    def load_from_arrays(
        x: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        val_split: float = 0.2,
        test_split: float = 0.0,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> Dict[str, Any]:
        """Split tensors into train/val/test splits without data leakage."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        if isinstance(y, np.ndarray):
            y = torch.tensor(y, dtype=torch.long if np.issubdtype(y.dtype, np.integer) else torch.float32)

        n = len(x)
        perm = torch.randperm(n)
        x_shuffled = x[perm]
        y_shuffled = y[perm]

        n_test = int(test_split * n)
        n_val = int(val_split * n)
        n_train = n - n_val - n_test

        train_ds = OmegaDataset(x_shuffled[:n_train], y_shuffled[:n_train])
        val_ds = OmegaDataset(x_shuffled[n_train : n_train + n_val], y_shuffled[n_train : n_train + n_val])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        test_loader = None
        if n_test > 0:
            test_ds = OmegaDataset(x_shuffled[n_train + n_val :], y_shuffled[n_train + n_val :])
            test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds) if test_loader else 0,
            "input_dim": x.shape[1] if x.ndim > 1 else 1,
            "output_dim": int(y.max().item() + 1) if y.dtype == torch.long else (y.shape[1] if y.ndim > 1 else 1),
        }
