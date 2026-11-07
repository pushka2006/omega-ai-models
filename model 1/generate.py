"""Omega GPT-1 Generation Interface (Model 1).

Generates text completions from trained checkpoints using temperature scaling,
top-k filtering, nucleus (top-p) sampling, and repetition penalty.
Supports CLI flags, prompt generation, real-time streaming, and interactive chat.
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F

from config import block_size, checkpoint_path, device
from model import GPTLanguageModel
from tokenizer import OmegaTokenizer, get_tokenizer


def load_model(path: str = checkpoint_path) -> GPTLanguageModel:
    """Load GPTLanguageModel with trained weights."""
    model = GPTLanguageModel().to(device)

    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        path,
        os.path.join(base, "checkpoints", "omega_gpt1_best.pth"),
        os.path.join(base, "checkpoints", "omega_gpt1_colab.pth"),
        os.path.join(base, "omega-gpt", "outputs", "checkpoints", "omega_gpt1_best.pth"),
        os.path.join(base, "checkpoints", "gpt_model.pth"),
        "gpt_model.pth",
    ]

    loaded = False
    for candidate in candidates:
        if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
            try:
                ckpt = torch.load(candidate, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    state_dict = ckpt["model_state_dict"]
                    step_info = f" (Step {ckpt.get('step', '?')}, Val Loss {ckpt.get('val_loss', '?')})"
                elif isinstance(ckpt, dict) and not any(k.startswith("embeddings") or k.startswith("blocks") for k in ckpt.keys()):
                    continue
                else:
                    state_dict = ckpt
                    step_info = ""

                # Load with strict=False to gracefully adapt if minor buffer differences exist
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded weights from {candidate}{step_info}")
                loaded = True
                break
            except Exception as e:
                # Try next candidate
                continue

    if not loaded:
        print("Warning: No valid checkpoint found. Generating with uninitialized weights.")

    model.eval()
    return model


def generate_text(
    prompt: str = "The future of AI is",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stream: bool = False,
) -> str:
    """Generate text completion from a given prompt."""
    tokenizer = get_tokenizer()
    model = load_model()

    idx = tokenizer.encode_to_tensor(prompt, device=device)
    if idx.shape[1] == 0:
        idx = torch.tensor([[tokenizer.bos_id if hasattr(tokenizer, 'bos_id') else 0]], dtype=torch.long, device=device)

    total_tokens = 0
    t0 = time.time()

    if not stream:
        output_ids = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        generated_list = output_ids[0].tolist()
        return tokenizer.decode(generated_list)

    # Real-time token-by-token streaming
    sys.stdout.write(prompt)
    sys.stdout.flush()

    curr_idx = idx
    for _ in range(max_new_tokens):
        idx_cond = curr_idx if curr_idx.size(1) <= block_size else curr_idx[:, -block_size:]
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        # Repetition penalty
        if repetition_penalty != 1.0:
            for b in range(curr_idx.size(0)):
                seen = set(curr_idx[b].tolist())
                for tid in seen:
                    if tid < logits.size(-1):
                        if logits[b, tid] > 0:
                            logits[b, tid] /= repetition_penalty
                        else:
                            logits[b, tid] *= repetition_penalty

        # Temperature
        if temperature > 0.0:
            logits = logits / max(temperature, 1e-6)

            if top_k > 0:
                threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)[0][:, -1, None]
                logits[logits < threshold] = float("-inf")

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                for b in range(logits.size(0)):
                    logits[b, sorted_indices[b][sorted_mask[b]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_token.item()
        curr_idx = torch.cat((curr_idx, next_token), dim=1)
        total_tokens += 1

        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        sys.stdout.write(piece)
        sys.stdout.flush()

        if hasattr(tokenizer, 'eos_id') and token_id == tokenizer.eos_id:
            break

    elapsed = time.time() - t0
    sys.stdout.write("\n")
    speed = total_tokens / elapsed if elapsed > 0 else 0
    print(f"[{total_tokens} tokens generated in {elapsed:.2f}s | {speed:.1f} tok/s]")
    return tokenizer.decode(curr_idx[0].tolist())


def interactive_mode():
    """Launch interactive real-time generation console."""
    print("=" * 60)
    print("       OMEGA GPT-1 INTERACTIVE GENERATION CONSOLE")
    print("=" * 60)
    print("Type your prompt and press Enter. Commands:")
    print("  /temp <float>   Set temperature (default: 0.7)")
    print("  /tokens <int>   Set max tokens (default: 100)")
    print("  /topk <int>     Set top-k (default: 50)")
    print("  /topp <float>   Set top-p (default: 0.9)")
    print("  /exit           Quit interactive console")
    print("-" * 60)

    temperature = 0.7
    max_tokens = 100
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1

    while True:
        try:
            prompt = input("\nOmega-GPT > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not prompt:
            continue

        if prompt.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Session ended.")
            break

        if prompt.startswith("/temp "):
            try:
                temperature = float(prompt.split()[1])
                print(f"[Temperature updated to {temperature}]")
            except ValueError:
                print("Invalid temperature value.")
            continue

        if prompt.startswith("/tokens "):
            try:
                max_tokens = int(prompt.split()[1])
                print(f"[Max tokens updated to {max_tokens}]")
            except ValueError:
                print("Invalid token count.")
            continue

        if prompt.startswith("/topk "):
            try:
                top_k = int(prompt.split()[1])
                print(f"[Top-k updated to {top_k}]")
            except ValueError:
                print("Invalid top-k value.")
            continue

        if prompt.startswith("/topp "):
            try:
                top_p = float(prompt.split()[1])
                print(f"[Top-p updated to {top_p}]")
            except ValueError:
                print("Invalid top-p value.")
            continue

        print("\n--- Output ---")
        generate_text(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omega GPT-1 Text Generation")
    parser.add_argument("--prompt", type=str, default="The future of AI is", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=80, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--topk", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--topp", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--repetition", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--stream", action="store_true", default=True, help="Stream tokens in real time")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive chat mode")

    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    else:
        print(f"Prompt: {args.prompt}")
        print("-" * 50)
        generate_text(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.topk,
            top_p=args.topp,
            repetition_penalty=args.repetition,
            stream=args.stream,
        )
