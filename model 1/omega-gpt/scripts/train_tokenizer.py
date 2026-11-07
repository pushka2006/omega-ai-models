"""Script: Train BPE Tokenizer for Omega GPT-1."""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizer.train_tokenizer import train_bpe_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--save_dir", type=str, default="tokenizer/saved")
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.data_dir, "*.txt")) + glob.glob(os.path.join(args.data_dir, "**", "*.txt"))
    files = sorted(list(set(files)))

    if not files:
        print(f"Error: No .txt files found in {args.data_dir}")
        sys.exit(1)

    train_bpe_tokenizer(files=files, vocab_size=args.vocab_size, save_dir=args.save_dir)


if __name__ == "__main__":
    main()

