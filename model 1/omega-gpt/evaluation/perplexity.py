"""Perplexity calculation utility for Omega GPT-1."""

import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from config.model_config import ModelConfig
from model.gpt import OmegaGPT
from tokenizer.tokenizer_utils import OmegaTokenizer
from training.dataset import OmegaDataset


@torch.no_grad()
def calculate_dataset_perplexity(
    checkpoint_path: str = "outputs/checkpoints/omega_gpt1_best.pth",
    tokens_path: str = "data/processed/val_tokens.pt",
    batch_size: int = 16,
    device: str = "auto",
) -> float:
    """Calculate exact perplexity over a tokenized dataset."""
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"Dataset tokens not found at: {tokens_path}")

    print(f"[Perplexity] Loading checkpoint from '{checkpoint_path}'...")
    ckpt = torch.load(checkpoint_path, map_location=dev)
    config = ModelConfig.from_dict(ckpt["model_config"])

    model = OmegaGPT(config).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokens = torch.load(tokens_path, weights_only=True)
    dataset = OmegaDataset(tokens, config.context_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_tokens = 0

    print(f"[Perplexity] Evaluating over {len(dataset):,} context chunks on {dev}...")
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item() * x.size(0) * x.size(1)
            total_tokens += x.size(0) * x.size(1)

    avg_nll = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_nll, 20.0))

    print("=" * 50)
    print("OMEGA GPT-1 PERPLEXITY REPORT")
    print(f"Total evaluated tokens : {total_tokens:,}")
    print(f"Average Negative Log-Likelihood : {avg_nll:.4f}")
    print(f"Perplexity (PPL)       : {ppl:.2f}")
    print("=" * 50)

    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for Omega GPT-1")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/omega_gpt1_best.pth")
    parser.add_argument("--tokens", type=str, default="data/processed/val_tokens.pt")
    args = parser.parse_args()

    calculate_dataset_perplexity(checkpoint_path=args.checkpoint, tokens_path=args.tokens)

