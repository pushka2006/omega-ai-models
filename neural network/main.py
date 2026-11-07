"""OMEGA NEURAL NETWORK ENGINE: Main CLI and Runtime Entrypoint.

Commands:
  python main.py info               Inspect hardware, devices, and framework capabilities
  python main.py train              Run training on benchmark dataset
  python main.py test               Execute comprehensive automated test suite
  python main.py dashboard          Launch the futuristic visualization and training dashboard
"""

import argparse
import sys
import torch
import uvicorn

from core.tensor_engine import tensor_engine
from datasets.dataset_loader import DatasetLoader, SyntheticDatasetGenerator
from evaluation.evaluator import Evaluator
from models.mlp import MLP
from optimizers.optimizer_factory import configure_optimizers
from training.trainer import TrainingEngine


def command_info():
    """Print hardware and framework diagnostics."""
    print("=" * 65)
    print("           OMEGA NEURAL NETWORK ENGINE DIAGNOSTICS")
    print("=" * 65)
    info = tensor_engine.get_device_info()
    print(f"  PyTorch Version      : {info['pytorch_version']}")
    print(f"  Active Compute Device: {info['active_device'].upper()}")
    print(f"  Hardware Device      : {info['gpu_name']}")
    print(f"  CUDA Available       : {info['cuda_available']}")
    print(f"  MPS Available        : {info['mps_available']}")
    if info.get("total_vram_gb"):
        print(f"  Total VRAM           : {info['total_vram_gb']} GB")
        print(f"  Allocated VRAM       : {info['allocated_vram_gb']} GB")
    print("\n  Supported Layers     : Dense, Conv1D, Conv2D, ConvTranspose2D,")
    print("                         MaxPool, AvgPool, BatchNorm, LayerNorm, RMSNorm,")
    print("                         Dropout, RNN, LSTM, GRU, MultiHeadAttention, TransformerBlock")
    print("  Supported Optimizers : AdamW, Adam, SGD, RMSprop, Adagrad")
    print("  Supported Activations: ReLU, LeakyReLU, GELU, SiLU/Swish, Sigmoid, Tanh, Softmax, ELU")
    print("=" * 65)


def command_train(args):
    """Run model training directly from terminal."""
    if args.task == "regression":
        x, y = SyntheticDatasetGenerator.generate_linear_regression(num_samples=1000, in_features=args.input_dim)
        criterion = torch.nn.MSELoss()
        task_type = "regression"
    else:
        x, y = SyntheticDatasetGenerator.generate_spiral(samples_per_class=args.samples_per_class, num_classes=args.classes)
        criterion = torch.nn.CrossEntropyLoss()
        task_type = "classification"

    data = DatasetLoader.load_from_arrays(x, y, val_split=0.2, batch_size=args.batch_size)
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")]

    model = MLP(
        input_dim=data["input_dim"],
        hidden_dims=hidden_dims,
        output_dim=data["output_dim"],
        activation=args.activation,
        use_batch_norm=args.batch_norm,
        dropout=args.dropout,
    )

    optimizer = configure_optimizers(model, optimizer_name=args.optimizer, learning_rate=args.lr)
    trainer = TrainingEngine(model=model, criterion=criterion, optimizer=optimizer)

    results = trainer.fit(
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        epochs=args.epochs,
        task_type=task_type,
        checkpoint_name=f"cli_{args.task}.pth",
    )
    print(f"\nFinal Result: Best Validation Metric = {results['best_metric']:.4f}")


def command_dashboard(args):
    """Launch the interactive dashboard server."""
    print("=" * 65)
    print("     LAUNCHING OMEGA NEURAL NETWORK ENGINE DASHBOARD")
    print("=" * 65)
    print(f"Dashboard UI: http://{args.host}:{args.port}")
    print(f"API Docs    : http://{args.host}:{args.port}/docs")
    uvicorn.run("dashboard.server:app", host=args.host, port=args.port, reload=args.reload)


def main():
    parser = argparse.ArgumentParser(description="OMEGA Neural Network Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Info
    subparsers.add_parser("info", help="Display framework and hardware diagnostics")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a neural network model")
    train_parser.add_argument("--task", type=str, default="spiral", choices=["spiral", "regression"])
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument("--optimizer", type=str, default="adamw")
    train_parser.add_argument("--activation", type=str, default="relu")
    train_parser.add_argument("--hidden-dims", type=str, default="64,32,16")
    train_parser.add_argument("--batch-norm", action="store_true", default=True)
    train_parser.add_argument("--dropout", type=float, default=0.1)
    train_parser.add_argument("--samples-per-class", type=int, default=350)
    train_parser.add_argument("--classes", type=int, default=3)
    train_parser.add_argument("--input-dim", type=int, default=10)

    # Test
    subparsers.add_parser("test", help="Run comprehensive automated test suite")

    # Dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Start web dashboard server")
    dash_parser.add_argument("--host", type=str, default="127.0.0.1")
    dash_parser.add_argument("--port", type=int, default=8500)
    dash_parser.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        command_info()
    elif args.command == "train":
        command_train(args)
    elif args.command == "test":
        import unittest
        suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
        runner = unittest.TextTestRunner(verbosity=2)
        runner.run(suite)
    elif args.command == "dashboard":
        command_dashboard(args)


if __name__ == "__main__":
    main()
