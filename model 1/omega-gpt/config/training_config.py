"""Training Configuration for Omega GPT-1."""

from dataclasses import asdict, dataclass
import os
from typing import Any, Dict


@dataclass
class TrainingConfig:
    # Optimization
    batch_size: int = 32
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 5000
    warmup_steps: int = 200
    decay_steps: int = 5000

    # Iteration & Validation
    eval_interval: int = 250
    eval_steps: int = 20
    save_interval: int = 500
    log_interval: int = 10
    generate_interval: int = 250

    # Compute
    device: str = "auto"  # "auto", "cuda", "cpu"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    compile_model: bool = False
    num_workers: int = 0
    seed: int = 42

    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    tokenizer_dir: str = "tokenizer/saved"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    sample_dir: str = "outputs/samples"
    graph_dir: str = "outputs/graphs"

    # Checkpoint Filenames
    latest_checkpoint_name: str = "omega_gpt1_latest.pth"
    best_checkpoint_name: str = "omega_gpt1_best.pth"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def get_training_config(**overrides) -> TrainingConfig:
    config = TrainingConfig(**overrides)
    return config

