"""Polished interactive terminal console for Omega GPT-1."""

import os
import sys
from model.model_utils import format_parameter_count
from .generate import TextGenerator


def run_interactive_session(
    checkpoint_path: str = None,
    tokenizer_dir: str = "tokenizer/saved",
    device: str = "auto",
):
    """Run interactive text generation console with live streaming and parameter tuning."""
    try:
        generator = TextGenerator(
            checkpoint_path=checkpoint_path,
            tokenizer_dir=tokenizer_dir,
            device=device,
        )
    except Exception as e:
        print(f"Error initializing Omega GPT-1: {e}")
        return

    val_loss_str = (
        f"{generator.checkpoint_info['val_loss']:.4f}"
        if generator.checkpoint_info.get("val_loss") is not None
        else "N/A"
    )

    print("=" * 60)
    print("                      OMEGA GPT-1")
    print("             CUSTOM LANGUAGE MODEL ENGINE")
    print("=" * 60)
    print(f"Model Name        : Omega GPT-1 (Decoder-Only Transformer)")
    print(f"Total Parameters  : {generator.total_params:,} ({format_parameter_count(generator.total_params)})")
    print(f"Vocabulary Size   : {generator.model_config.vocab_size:,} tokens")
    print(f"Context Length    : {generator.model_config.context_length} tokens")
    print(f"Layers / Heads    : {generator.model_config.num_layers} layers / {generator.model_config.num_heads} heads")
    print(f"Compute Device    : {generator.device.upper()}")
    print(f"Active Checkpoint : {generator.checkpoint_info['path']}")
    print(f"Training Step     : {generator.checkpoint_info['step']} (Val Loss: {val_loss_str})")
    print("=" * 60)
    print("Commands:")
    print("  /temp <float>       Set temperature (default: 0.8)")
    print("  /tokens <int>       Set max generated tokens (default: 120)")
    print("  /topk <int>         Set Top-K filtering (default: 40)")
    print("  /topp <float>       Set Top-P nucleus threshold (default: 0.9)")
    print("  /repetition <float> Set repetition penalty (default: 1.1)")
    print("  /reset              Reset settings to default")
    print("  exit / quit         Exit the console")
    print("-" * 60)

    temperature = 0.8
    max_tokens = 120
    top_k = 40
    top_p = 0.9
    repetition_penalty = 1.1

    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if not prompt:
                continue

            if prompt.lower() in ("exit", "quit"):
                print("\nExit: Closing Omega GPT-1 console. Goodbye!")
                break

            # Command handling
            if prompt.startswith("/temp"):
                parts = prompt.split()
                if len(parts) > 1:
                    temperature = max(0.01, float(parts[1]))
                    print(f"-> Temperature set to {temperature:.2f}")
                continue
            elif prompt.startswith("/tokens"):
                parts = prompt.split()
                if len(parts) > 1:
                    max_tokens = max(1, int(parts[1]))
                    print(f"-> Max tokens set to {max_tokens}")
                continue
            elif prompt.startswith("/topk"):
                parts = prompt.split()
                if len(parts) > 1:
                    top_k = int(parts[1])
                    print(f"-> Top-K set to {top_k}")
                continue
            elif prompt.startswith("/topp"):
                parts = prompt.split()
                if len(parts) > 1:
                    top_p = float(parts[1])
                    print(f"-> Top-P set to {top_p:.2f}")
                continue
            elif prompt.startswith("/repetition"):
                parts = prompt.split()
                if len(parts) > 1:
                    repetition_penalty = float(parts[1])
                    print(f"-> Repetition penalty set to {repetition_penalty:.2f}")
                continue
            elif prompt == "/reset":
                temperature = 0.8
                max_tokens = 120
                top_k = 40
                top_p = 0.9
                repetition_penalty = 1.1
                print("-> Settings reset to defaults.")
                continue

            print("\nGenerate text:")
            print("-" * 50)
            print(prompt, end="", flush=True)

            stream = generator.generate_stream(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            for chunk in stream:
                print(chunk, end="", flush=True)

            print("\n" + "-" * 50)

        except (KeyboardInterrupt, EOFError):
            print("\nExit: Console session closed.")
            break


if __name__ == "__main__":
    run_interactive_session()
