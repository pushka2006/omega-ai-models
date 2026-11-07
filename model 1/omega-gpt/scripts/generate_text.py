"""Script: Generate text from prompt with Omega GPT-1."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from generation.generate import TextGenerator
from generation.interactive_chat import run_interactive_session


def main():
    parser = argparse.ArgumentParser(description="Generate text using Omega GPT-1")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to complete")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument("--tokens", type=int, default=120, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P nucleus sampling")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive console session")
    args = parser.parse_args()

    if args.interactive or args.prompt is None:
        run_interactive_session(checkpoint_path=args.checkpoint)
    else:
        gen = TextGenerator(checkpoint_path=args.checkpoint)
        res = gen.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("\n" + "=" * 50)
        print("OMEGA GPT-1 GENERATED OUTPUT:")
        print("=" * 50)
        print(res["generated_text"])
        print("-" * 50)
        print(f"Speed: {res['tokens_per_second']:.1f} tokens/sec ({res['tokens_generated']} tokens)")


if __name__ == "__main__":
    main()

