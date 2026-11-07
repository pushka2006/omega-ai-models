"""OMEGA NEURAL NETWORK ENGINE: Tensor and Compute Engine.

Handles device allocation (CUDA, Apple MPS, CPU), memory monitoring,
reproducible seeding, and tensor profiling.
"""

import os
import random
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch


class TensorEngine:
    """Manages computation devices, memory profiling, and tensor operations."""

    @staticmethod
    def get_device(preferred: Optional[str] = None) -> torch.device:
        """Detect and return the optimal compute device."""
        if preferred:
            pref_lower = preferred.lower()
            if pref_lower.startswith("cuda") and torch.cuda.is_available():
                return torch.device(pref_lower)
            elif pref_lower == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Return comprehensive hardware acceleration statistics."""
        cuda_avail = torch.cuda.is_available()
        mps_avail = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        info = {
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "mps_available": mps_avail,
            "active_device": "cuda" if cuda_avail else ("mps" if mps_avail else "cpu"),
            "gpu_count": torch.cuda.device_count() if cuda_avail else 0,
        }

        if cuda_avail:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            alloc_mem = torch.cuda.memory_allocated(0) / (1024**3)
            info["total_vram_gb"] = round(total_mem, 2)
            info["allocated_vram_gb"] = round(alloc_mem, 2)
        else:
            info["gpu_name"] = "CPU Execution Engine"

        return info

    @staticmethod
    def set_seed(seed: int = 42, deterministic: bool = True):
        """Set deterministic random seeds across Python, NumPy, and PyTorch."""
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic and torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
        """Return (total_params, trainable_params)."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    @staticmethod
    def get_model_size_mb(model: torch.nn.Module) -> float:
        """Estimate model memory footprint in megabytes."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return round((param_size + buffer_size) / (1024**2), 2)


# Singleton
tensor_engine = TensorEngine()
