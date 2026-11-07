"""OMEGA NEURAL NETWORK ENGINE: Configuration.

Manages framework directories, device defaults, and experiment tracking settings.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR.parent
MODEL1_DIR = WORKSPACE_DIR / "model 1"
MODEL2_DIR = WORKSPACE_DIR / "model 2 ai assistant"

# Storage & Output Paths
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_DIR = BASE_DIR / "checkpoints_saved"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_DB_PATH = BASE_DIR / "experiments" / "experiments.db"
EXPERIMENT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Default Compute & Training Hyperparameters
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
