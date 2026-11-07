"""Configuration module for OMEGA NEXUS MODEL (Model 4).

Coordinates cross-model paths, compute device, and fusion dimensions.
"""

from pathlib import Path
import torch

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parent

MODEL_1_DIR = WORKSPACE_ROOT / "model 1"
MODEL_2_DIR = WORKSPACE_ROOT / "model 2 ai assistant"
MODEL_3_DIR = WORKSPACE_ROOT / "neural network"

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

# Nexus Architecture Hyperparameters
LATENT_DIM = 256
NUM_FUSION_HEADS = 4
DROPOUT = 0.1
VERSION = "4.0.0-nexus"

