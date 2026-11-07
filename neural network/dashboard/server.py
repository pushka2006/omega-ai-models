"""OMEGA NEURAL NETWORK ENGINE: Dashboard Server.

FastAPI application serving REST endpoints and the interactive training dashboard.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from builder.model_builder import model_builder
from checkpoints.checkpoint_manager import checkpoint_manager
from core.tensor_engine import tensor_engine
from datasets.dataset_loader import DatasetLoader, SyntheticDatasetGenerator
from evaluation.evaluator import Evaluator
from experiments.tracker import experiment_tracker
from models.mlp import MLP
from optimizers.optimizer_factory import configure_optimizers
from training.trainer import TrainingEngine
from visualization.visualizer import ModelVisualizer

app = FastAPI(title="OMEGA Neural Network Engine", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "OMEGA Neural Network Engine Dashboard"}


@app.get("/api/status")
def get_system_status():
    """Return hardware compute status, device memory, and engine health."""
    device_info = tensor_engine.get_device_info()
    experiments = experiment_tracker.list_experiments(limit=5)
    return {
        "engine": "OMEGA Neural Network Engine v2.0",
        "hardware": device_info,
        "recent_experiments_count": len(experiments),
    }


class ModelBuildRequest(BaseModel):
    name: str = "custom_net"
    input_size: int = 10
    layers: List[Dict[str, Any]]


@app.post("/api/builder/validate")
def validate_model_config(req: ModelBuildRequest):
    """Validate declarative model specification and return layer summary."""
    try:
        model = model_builder.build_from_config(req.dict())
        summary = ModelVisualizer.get_layer_summary(model)
        total_p, train_p = tensor_engine.count_parameters(model)
        size_mb = tensor_engine.get_model_size_mb(model)
        return {
            "valid": True,
            "model_name": req.name,
            "total_parameters": total_p,
            "trainable_parameters": train_p,
            "size_mb": size_mb,
            "layers": summary,
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


class TrainRequest(BaseModel):
    task: str = "spiral_classification"  # 'spiral_classification' or 'regression'
    hidden_dims: List[int] = [64, 32, 16]
    epochs: int = 30
    learning_rate: float = 0.005
    optimizer: str = "adamw"
    activation: str = "relu"
    batch_size: int = 32


@app.post("/api/train/start")
def start_training_run(req: TrainRequest):
    """Execute a real, verifiable training run and persist the experiment."""
    # 1. Generate Dataset
    if req.task == "regression":
        x, y = SyntheticDatasetGenerator.generate_linear_regression(num_samples=800, in_features=10)
        criterion = torch.nn.MSELoss()
        task_type = "regression"
    else:
        x, y = SyntheticDatasetGenerator.generate_spiral(samples_per_class=350, num_classes=3)
        criterion = torch.nn.CrossEntropyLoss()
        task_type = "classification"

    data = DatasetLoader.load_from_arrays(x, y, val_split=0.2, batch_size=req.batch_size)

    # 2. Instantiate Model
    model = MLP(
        input_dim=data["input_dim"],
        hidden_dims=req.hidden_dims,
        output_dim=data["output_dim"],
        activation=req.activation,
        use_batch_norm=True,
        dropout=0.1,
    )

    optimizer = configure_optimizers(
        model=model,
        optimizer_name=req.optimizer,
        learning_rate=req.learning_rate,
    )

    # 3. Train
    trainer = TrainingEngine(model=model, criterion=criterion, optimizer=optimizer)
    results = trainer.fit(
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        epochs=req.epochs,
        task_type=task_type,
        checkpoint_name=f"dashboard_{req.task}.pth",
    )

    # 4. Log Experiment
    exp_id = experiment_tracker.log_run(
        name=f"Dashboard-{req.task.capitalize()}",
        model_config={"hidden_dims": req.hidden_dims, "activation": req.activation},
        dataset_info={"train_samples": data["train_size"], "val_samples": data["val_size"]},
        hyperparameters={"epochs": req.epochs, "lr": req.learning_rate, "optimizer": req.optimizer},
        best_val_metric=results["best_metric"],
        final_train_loss=results["history"][-1]["train_loss"] if results["history"] else 0.0,
        epochs_trained=results["epochs_trained"],
        duration_seconds=results["training_time_seconds"],
        checkpoint_path=f"dashboard_{req.task}.pth",
    )

    return {
        "experiment_id": exp_id,
        "results": results,
        "layer_summary": ModelVisualizer.get_layer_summary(model),
        "weight_stats": ModelVisualizer.inspect_weight_distributions(model),
    }


@app.get("/api/experiments")
def list_experiments():
    return experiment_tracker.list_experiments(limit=30)


@app.get("/api/checkpoints")
def list_saved_checkpoints():
    from config.engine_config import CHECKPOINT_DIR
    ckpts = []
    if os.path.exists(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith(".pth"):
                full = os.path.join(CHECKPOINT_DIR, f)
                ckpts.append({
                    "filename": f,
                    "size_mb": round(os.path.getsize(full) / (1024**2), 2),
                    "modified": os.path.getmtime(full),
                })
    return ckpts
