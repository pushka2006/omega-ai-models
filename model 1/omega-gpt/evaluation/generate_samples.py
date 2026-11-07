"""Generate structured benchmark sample outputs for qualitative evaluation."""

import os
from typing import List
from generation.generate import TextGenerator


def generate_benchmark_samples(
    checkpoint_path: str = "outputs/checkpoints/omega_gpt1_best.pth",
    output_file: str = "outputs/samples/benchmark_eval.txt",
    prompts: List[str] = None,
):
    """Run generation across standard benchmark prompts and varied temperatures."""
    if prompts is None:
        prompts = [
            "Artificial intelligence is",
            "The Transformer architecture uses",
            "In language modeling, next-token prediction",
            "Self-attention allows the model to",
        ]

    generator = TextGenerator(checkpoint_path=checkpoint_path)
    temperatures = [0.2, 0.7, 1.0]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("OMEGA GPT-1 BENCHMARK EVALUATION SAMPLES\n")
        f.write("=" * 60 + "\n\n")

        for p_idx, prompt in enumerate(prompts, 1):
            f.write(f"PROMPT {p_idx}: \"{prompt}\"\n")
            f.write("-" * 50 + "\n")
            print(f"\nEvaluating Prompt {p_idx}: '{prompt}'")

            for temp in temperatures:
                res = generator.generate(
                    prompt=prompt,
                    max_new_tokens=80,
                    temperature=temp,
                    top_k=40,
                    top_p=0.9,
                )
                f.write(f"[Temperature = {temp:.1f} | Speed: {res['tokens_per_second']:.1f} tok/s]:\n")
                f.write(res["generated_text"] + "\n\n")
                print(f"  [T={temp:.1f}] {res['generated_text'][:80]}...")

            f.write("=" * 60 + "\n\n")

    print(f"\n[Evaluation] Benchmark samples saved to '{output_file}'")


if __name__ == "__main__":
    generate_benchmark_samples()

