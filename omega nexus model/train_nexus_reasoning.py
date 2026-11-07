"""OMEGA NEXUS MODEL: Train Multimodal Cross-Attention Fusion on Fable Reasoning Dataset.

Fuses text embeddings with continuous neural feature vectors using cross-attention,
training the orchestrator to calibrate reasoning complexity and route execution.
"""

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import LATENT_DIM, DEVICE
from core.fusion_layer import NexusFusionNetwork
from adapters.gpt1_adapter import GPT1Adapter

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets_hub" / "processed" / "fable_reasoning_nexus.pt"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = CKPT_DIR / "nexus_reasoning_fusion.pth"


def train_nexus_fusion(epochs: int = 25, batch_size: int = 16, lr: float = 0.005):
    print("=" * 65)
    print("  OMEGA NEXUS MODEL: MULTIMODAL REASONING FUSION TRAINING")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Nexus dataset missing at '{DATA_PATH}'.")

    data = torch.load(str(DATA_PATH), map_location="cpu", weights_only=False)
    prompts = data["prompts"]
    features = data["features"]
    labels = data["labels"]

    print(f"Loaded {len(prompts)} multimodal pairs. Feature shape: {features.shape}.")

    # Extract text embeddings via Model 1 adapter
    gpt1 = GPT1Adapter()
    print("[Nexus] Precomputing continuous text representations via OMEGA GPT-1...")

    text_embs = []
    for p in prompts:
        emb = gpt1.get_token_embeddings(p[:100])
        # Pool to fixed length seq_len=4 or mean pool
        if emb.size(1) > 8:
            emb = emb[:, :8, :]
        elif emb.size(1) < 8:
            emb = torch.nn.functional.pad(emb, (0, 0, 0, 8 - emb.size(1)))
        text_embs.append(emb[0])

    text_tensor = torch.stack(text_embs)
    print(f"Text Embeddings shape: {text_tensor.shape} | Features shape: {features.shape}")

    # Train / Val Split (80% / 20%)
    indices = torch.randperm(len(text_tensor))
    split = int(0.8 * len(text_tensor))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(text_tensor[train_idx], features[train_idx], labels[train_idx])
    val_ds = TensorDataset(text_tensor[val_idx], features[val_idx], labels[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = NexusFusionNetwork(
        text_dim=256,
        feature_dim=16,
        latent_dim=LATENT_DIM,
        num_classes=4,
        num_heads=4,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for t_b, f_b, y_b in train_loader:
            t_b, f_b, y_b = t_b.to(DEVICE), f_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            _, logits = model(t_b, f_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        train_loss /= total
        train_acc = (correct / total) * 100.0

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for t_b, f_b, y_b in val_loader:
                t_b, f_b, y_b = t_b.to(DEVICE), f_b.to(DEVICE), y_b.to(DEVICE)
                _, logits = model(t_b, f_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_loss /= max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "latent_dim": LATENT_DIM,
                "val_accuracy": val_acc,
                "epoch": epoch,
            }, SAVE_PATH)

        star = " ★ [NEW BEST]" if is_best else ""
        if epoch % 5 == 0 or epoch == epochs or is_best:
            print(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%{star}"
            )

    print("-" * 65)
    print(f"Nexus Multimodal Training Complete! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Fusion weights saved to: '{SAVE_PATH}'")
    print("=" * 65)
    return best_val_acc


if __name__ == "__main__":
    train_nexus_fusion()
