"""Omega GPT-1 Production Training Pipeline (Model 1).

Features:
- Byte Pair Encoding (BPE) subword tokenization
- Strict 90/10 train/validation split without data leakage
- AdamW optimizer with decoupled weight decay for 2D matrices
- Cosine learning rate scheduling with linear warmup
- Gradient norm clipping
- Periodic validation loss and perplexity evaluation
- Best and latest checkpoint persistence with recovery metadata
"""

import argparse
import glob
import math
import os
import sys
import time
import torch
import torch.nn.functional as F

from config import (
    batch_size as default_batch_size,
    betas,
    block_size,
    checkpoint_dir,
    checkpoint_path as default_checkpoint_path,
    data_path,
    device,
    eval_interval,
    eval_iters,
    grad_clip,
    learning_rate as default_lr,
    max_iters as default_max_iters,
    min_lr,
    raw_data_dir,
    warmup_iters,
    weight_decay,
)
from model import GPTLanguageModel
from tokenizer import get_tokenizer


def load_raw_text(primary_path: str, raw_dir: str) -> str:
    """Load and aggregate text from primary training file and any raw corpus files."""
    texts = []
    if os.path.exists(primary_path) and os.path.getsize(primary_path) > 0:
        with open(primary_path, "r", encoding="utf-8") as f:
            texts.append(f.read())

    if os.path.exists(raw_dir):
        for txt_file in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
            if os.path.abspath(txt_file) != os.path.abspath(primary_path) and os.path.getsize(txt_file) > 0:
                with open(txt_file, "r", encoding="utf-8") as f:
                    texts.append(f.read())

    combined = "\n\n".join(texts).strip()
    if not combined:
        raise ValueError(
            f"No training text found in '{primary_path}' or '{raw_dir}'. Please add training data first."
        )
    return combined


def get_lr(step: int, max_steps: int, peak_lr: float, min_learning_rate: float, warmup_steps: int) -> float:
    """Cosine learning rate schedule with preliminary linear warmup."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    if step > max_steps:
        return min_learning_rate
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (peak_lr - min_learning_rate)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    batch_sz: int,
    ctx_len: int,
    num_batches: int,
    dev: str,
) -> dict[str, float]:
    """Estimate average loss and perplexity across train and validation splits."""
    model.eval()
    out = {}
    splits = [("train", train_tokens), ("val", val_tokens)]

    for split_name, data_source in splits:
        losses = []
        n_avail = len(data_source) - ctx_len - 1
        if n_avail <= 0:
            continue
        eval_count = min(num_batches, max(1, n_avail // batch_sz))
        for _ in range(eval_count):
            ix = torch.randint(n_avail, (batch_sz,))
            x = torch.stack([data_source[i : i + ctx_len] for i in ix]).to(dev)
            y = torch.stack([data_source[i + 1 : i + ctx_len + 1] for i in ix]).to(dev)
            _, loss = model(x, y)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses) if losses else float("inf")
        out[split_name] = avg_loss
        out[f"{split_name}_ppl"] = math.exp(min(avg_loss, 20.0))

    model.train()
    return out


def train():
    parser = argparse.ArgumentParser(description="Train Omega GPT-1 from Scratch")
    parser.add_argument("--iters", type=int, default=default_max_iters, help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=default_lr, help="Peak learning rate")
    parser.add_argument("--eval-interval", type=int, default=eval_interval, help="Validation interval")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    args = parser.parse_args()

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, "omega_gpt1_best.pth")
    latest_ckpt_path = os.path.join(checkpoint_dir, "omega_gpt1_latest.pth")

    print("=" * 65)
    print("                 OMEGA GPT-1 TRAINING PIPELINE")
    print("=" * 65)
    print(f"Device: {device.upper()}")

    # 1. Load and Tokenize Corpus
    corpus_text = load_raw_text(data_path, raw_data_dir)
    print(f"Corpus Loaded: {len(corpus_text):,} characters")

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(corpus_text)
    print(f"Tokenized Corpus: {len(token_ids):,} subwords (Vocab size: {tokenizer.get_vocab_size():,})")

    if len(token_ids) <= block_size + 1:
        raise ValueError(
            f"Corpus token count ({len(token_ids)}) is insufficient for context length ({block_size})."
        )

    # 2. Train / Validation Split (90/10)
    split_idx = int(0.9 * len(token_ids))
    train_tokens = torch.tensor(token_ids[:split_idx], dtype=torch.long)
    val_tokens = torch.tensor(token_ids[split_idx:], dtype=torch.long)
    print(f"Train Split: {len(train_tokens):,} tokens | Val Split: {len(val_tokens):,} tokens")

    # 3. Model Initialization
    model = GPTLanguageModel().to(device)
    total_p, train_p = model.count_parameters()
    print(f"Model Parameters: {total_p:,} total | {train_p:,} trainable ({total_p/1e6:.2f}M)")

    # 4. Optimizer
    optimizer = model.configure_optimizers(learning_rate=args.lr, weight_decay=weight_decay, betas=betas)

    start_step = 0
    best_val_loss = float("inf")

    # Load existing best checkpoint benchmark to avoid overwriting superior models
    if os.path.exists(best_ckpt_path):
        try:
            prior_data = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(prior_data, dict) and "val_loss" in prior_data:
                best_val_loss = float(prior_data["val_loss"])
                print(f"Existing Best Checkpoint Benchmark: Val Loss = {best_val_loss:.4f}")
        except Exception:
            pass

    # 5. Checkpoint Resume
    if args.resume and os.path.exists(latest_ckpt_path):
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step} (Best Val Loss: {best_val_loss:.4f})")

    # 6. Batch Generation Function
    def get_batch(split: str):
        source = train_tokens if split == "train" else val_tokens
        ix = torch.randint(len(source) - block_size - 1, (args.batch_size,))
        x = torch.stack([source[i : i + block_size] for i in ix])
        y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)

    # 7. Training Loop
    print("-" * 65)
    print("Beginning Training Loop...")
    t_start = time.time()
    tokens_processed = 0

    model.train()
    for step in range(start_step, args.iters):
        t0 = time.time()

        # Dynamic Learning Rate
        lr = get_lr(step, args.iters, args.lr, min_lr, warmup_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Fetch batch & Forward pass
        xb, yb = get_batch("train")
        logits, loss = model(xb, yb)

        # Backward pass & Optimizer step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        dt = time.time() - t0
        tokens_processed += args.batch_size * block_size

        # Evaluation & Checkpoint Logging
        if step % args.eval_interval == 0 or step == args.iters - 1:
            eval_metrics = estimate_loss(
                model, train_tokens, val_tokens, args.batch_size, block_size, eval_iters, device
            )
            val_loss = eval_metrics.get("val", float("inf"))
            val_ppl = eval_metrics.get("val_ppl", float("inf"))
            t_elapsed = time.time() - t_start
            throughput = tokens_processed / t_elapsed if t_elapsed > 0 else 0

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            star = " ★ [NEW BEST]" if is_best else ""
            print(
                f"Step {step:4d}/{args.iters} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val PPL: {val_ppl:.2f} | "
                f"LR: {lr:.2e} | "
                f"{throughput:,.0f} tok/s{star}"
            )

            # Save Checkpoint
            checkpoint_payload = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "model_config": {
                    "vocab_size": model.vocab_size,
                    "context_length": model.block_size,
                    "d_model": model.n_embd,
                    "num_layers": model.n_layer,
                    "num_heads": model.n_head,
                    "d_ff": model.d_ff,
                    "dropout": model.dropout_rate,
                    "activation": model.activation,
                    "tie_word_embeddings": model.tie_weights,
                },
            }

            torch.save(checkpoint_payload, latest_ckpt_path)
            if is_best:
                torch.save(checkpoint_payload, best_ckpt_path)

    total_time = time.time() - t_start
    print("-" * 65)
    print(f"Training completed in {total_time:.2f} seconds!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: '{best_ckpt_path}' and '{latest_ckpt_path}'")

    # Sample Generation Test
    print("\n[SAMPLE GENERATION TEST]")
    model.eval()
    sample_prompt = "Artificial intelligence is"
    prompt_ids = tokenizer.encode_to_tensor(sample_prompt, device=device)
    with torch.no_grad():
        gen_ids = model.generate(prompt_ids, max_new_tokens=40, temperature=0.7)
    sample_out = tokenizer.decode(gen_ids[0].tolist())
    print(f"Output: {sample_out}")
    print("=" * 65)


if __name__ == "__main__":
    train()
