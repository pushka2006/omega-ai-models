"""OMEGA NEURAL NETWORK ENGINE: Data Preprocessor and Scalers."""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance (z-score)."""

    def __init__(self):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, x: Union[np.ndarray, torch.Tensor]) -> "StandardScaler":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.std = np.std(x, axis=0, keepdims=True)
        # Avoid division by zero on constant features
        self.std[self.std == 0.0] = 1.0
        return self

    def transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fitted before calling transform.")
        return (x - self.mean) / self.std

    def fit_transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.fit(x).transform(x)


class MinMaxScaler:
    """Scales features to a specified interval (default [0, 1])."""

    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self.feature_range = feature_range

    def fit(self, x: Union[np.ndarray, torch.Tensor]) -> "MinMaxScaler":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        self.min_val = np.min(x, axis=0, keepdims=True)
        self.max_val = np.max(x, axis=0, keepdims=True)
        diff = self.max_val - self.min_val
        diff[diff == 0.0] = 1.0
        self._diff = diff
        return self

    def transform(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        norm = (x - self.min_val) / self._diff
        return norm * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]


def compute_dataset_statistics(x: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    """Calculate summary statistics for a dataset."""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.array(x)
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.array(y)

    stats = {
        "num_samples": len(x_np),
        "feature_count": x_np.shape[1] if x_np.ndim > 1 else 1,
        "feature_means": [round(float(m), 4) for m in np.mean(x_np, axis=0)],
        "feature_stds": [round(float(s), 4) for s in np.std(x_np, axis=0)],
        "target_dtype": str(y_np.dtype),
    }

    if np.issubdtype(y_np.dtype, np.integer):
        unique, counts = np.unique(y_np, return_counts=True)
        stats["class_distribution"] = {int(k): int(v) for k, v in zip(unique, counts)}

    return stats
