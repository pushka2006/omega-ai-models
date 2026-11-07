"""Script: Prepare raw text data into train and val tokens."""

import argparse
import sys
import os

# Add parent directory to path so imports work smoothly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataset import prepare_and_tokenize_data


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for Omega GPT-1")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer/saved")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    args = parser.parse_args()

    prepare_and_tokenize_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        tokenizer_dir=args.tokenizer_dir,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()

