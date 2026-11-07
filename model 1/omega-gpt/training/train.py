"""Training launcher script for Omega GPT-1."""

import argparse
from config.model_config import get_model_config
from config.training_config import get_training_config
from tokenizer.tokenizer_utils import OmegaTokenizer
from .trainer import OmegaTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Omega GPT-1 language model from scratch")
    parser.add_argument("--profile", type=str, default="small", choices=["tiny", "small", "medium", "custom"])
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--resume", action="store_true", help="Resume from latest saved checkpoint")
    args = parser.parse_args()

    # Load configurations
    model_cfg = get_model_config(profile=args.profile)
    train_overrides = {"device": args.device}
    if args.max_steps is not None:
        train_overrides["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        train_overrides["learning_rate"] = args.learning_rate

    train_cfg = get_training_config(**train_overrides)

    # Initialize tokenizer and trainer
    tokenizer = OmegaTokenizer.load(train_cfg.tokenizer_dir)
    # Ensure model vocab matches tokenizer
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    trainer = OmegaTrainer(model_config=model_cfg, training_config=train_cfg, tokenizer=tokenizer)
    trainer.train(resume=args.resume)


if __name__ == "__main__":
    main()

