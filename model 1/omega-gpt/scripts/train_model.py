"""Script: Train Omega GPT-1 model from scratch."""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.model_config import get_model_config
from config.training_config import get_training_config
from tokenizer.tokenizer_utils import OmegaTokenizer
from training.trainer import OmegaTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Omega GPT-1 from scratch")
    parser.add_argument("--profile", type=str, default="small", choices=["tiny", "small", "medium", "custom"])
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model_cfg = get_model_config(profile=args.profile)
    overrides = {"device": args.device}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    train_cfg = get_training_config(**overrides)

    tokenizer = OmegaTokenizer.load(train_cfg.tokenizer_dir)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    trainer = OmegaTrainer(model_config=model_cfg, training_config=train_cfg, tokenizer=tokenizer)
    trainer.train(resume=False)


if __name__ == "__main__":
    main()

