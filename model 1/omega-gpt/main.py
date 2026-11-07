"""Main CLI Entrypoint for Omega GPT-1.

Supported Commands:
    python main.py info
    python main.py setup
    python main.py prepare-data
    python main.py train-tokenizer
    python main.py train
    python main.py resume
    python main.py generate
    python main.py evaluate
    python main.py test
"""

import argparse
import glob
import json
import os
import sys
import unittest

# Ensure current directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def command_info(args):
    """Display comprehensive architecture, tokenizer, checkpoint, and hardware diagnostics."""
    import torch
    from config.model_config import get_model_config
    from model.gpt import OmegaGPT
    from model.model_utils import count_parameters, format_parameter_count

    print("=" * 65)
    print("                 OMEGA GPT-1 SYSTEM DIAGNOSTICS")
    print("=" * 65)

    # 1. Hardware Environment
    print("[1] COMPUTE & HARDWARE ENVIRONMENT")
    print(f"  PyTorch Version      : {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA Available       : {cuda_avail}")
    if cuda_avail:
        print(f"  GPU Device           : {torch.cuda.get_device_name(0)}")
        print(f"  GPU Count            : {torch.cuda.device_count()}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM Total           : {vram_gb:.2f} GB")
    else:
        print("  Active Device Mode   : CPU (Fallback)")

    # 2. Model Architecture
    model_cfg = get_model_config(profile=args.profile)
    temp_model = OmegaGPT(model_cfg)
    total_p, train_p = count_parameters(temp_model)

    print("\n[2] MODEL ARCHITECTURE SPECIFICATIONS")
    print(f"  Active Profile       : {args.profile.upper()}")
    print(f"  Total Parameters     : {total_p:,} ({format_parameter_count(total_p)})")
    print(f"  Trainable Parameters : {train_p:,} ({format_parameter_count(train_p)})")
    print(f"  Transformer Layers   : {model_cfg.num_layers}")
    print(f"  Attention Heads      : {model_cfg.num_heads} (Head Dim: {model_cfg.head_dim})")
    print(f"  Embedding Dimension  : {model_cfg.d_model}")
    print(f"  Feed-Forward Dim     : {model_cfg.d_ff}")
    print(f"  Context Length       : {model_cfg.context_length} tokens")
    print(f"  Vocabulary Size      : {model_cfg.vocab_size:,} tokens")
    print(f"  Activation Function  : {model_cfg.activation.upper()}")
    print(f"  Weight Tying         : {model_cfg.tie_word_embeddings}")

    # 3. Tokenizer Status
    print("\n[3] TOKENIZER CONFIGURATION")
    tok_dir = "tokenizer/saved"
    tok_file = os.path.join(tok_dir, "tokenizer.json")
    if os.path.exists(tok_file):
        try:
            from tokenizer.tokenizer_utils import OmegaTokenizer
            tok = OmegaTokenizer.load(tok_dir)
            print(f"  Type                 : Byte Pair Encoding (BPE)")
            print(f"  Active Vocabulary    : {tok.get_vocab_size():,} subwords")
            print(f"  Special Tokens       : [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3")
            print(f"  Status               : Trained and Ready at '{tok_dir}'")
        except Exception as e:
            print(f"  Status               : Error reading tokenizer: {e}")
    else:
        print(f"  Status               : Not yet trained. Run 'python main.py train-tokenizer'")

    # 4. Checkpoints Status
    print("\n[4] SAVED CHECKPOINTS")
    ckpt_dir = "outputs/checkpoints"
    best_ckpt = os.path.join(ckpt_dir, "omega_gpt1_best.pth")
    latest_ckpt = os.path.join(ckpt_dir, "omega_gpt1_latest.pth")

    if os.path.exists(best_ckpt):
        try:
            ckpt_data = torch.load(best_ckpt, map_location="cpu", weights_only=False)
            print(f"  Best Checkpoint      : {best_ckpt}")
            print(f"    - Training Step    : {ckpt_data.get('step', '?')}")
            print(f"    - Validation Loss  : {ckpt_data.get('val_loss', '?'):.4f}")
        except Exception:
            print(f"  Best Checkpoint      : Present at {best_ckpt}")
    else:
        print("  Best Checkpoint      : None found")

    if os.path.exists(latest_ckpt):
        print(f"  Latest Checkpoint    : {latest_ckpt}")

    # 5. Dataset Statistics
    print("\n[5] DATASET PREPARATION METADATA")
    meta_path = "data/processed/dataset_metadata.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  Processed Documents  : {meta.get('num_documents', '?')}")
        print(f"  Raw Characters       : {meta.get('total_raw_chars', '?'):,}")
        print(f"  Total Tokens         : {meta.get('total_tokens', '?'):,}")
        print(f"  Training Split       : {meta.get('train_tokens', '?'):,} tokens")
        print(f"  Validation Split     : {meta.get('val_tokens', '?'):,} tokens")
    else:
        print("  Status               : Not yet preprocessed. Run 'python main.py prepare-data'")

    print("=" * 65)


def command_setup(args):
    """Setup project directory structure and verify environment."""
    print("=" * 60)
    print("SETTING UP OMEGA GPT-1 ENVIRONMENT")
    print("=" * 60)

    dirs = [
        "config",
        "data/raw",
        "data/processed",
        "tokenizer/saved",
        "model",
        "training",
        "generation",
        "evaluation",
        "scripts",
        "notebooks",
        "tests",
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/samples",
        "outputs/graphs",
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"  [OK] Verified directory: {d}")

    import torch
    print("-" * 60)
    print(f"PyTorch Version  : {torch.__version__}")
    print(f"CUDA Available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device       : {torch.cuda.get_device_name(0)}")
    else:
        print("Device Mode      : CPU (CUDA not active on this host)")
    print("=" * 60)
    print("Omega GPT-1 setup completed successfully.")


def command_train_tokenizer(args):
    """Train custom BPE tokenizer from raw corpus."""
    from tokenizer.train_tokenizer import train_bpe_tokenizer

    files = glob.glob(os.path.join(args.data_dir, "*.txt")) + glob.glob(os.path.join(args.data_dir, "**", "*.txt"))
    files = sorted(list(set(files)))
    if not files:
        print(f"Error: No .txt files found in '{args.data_dir}'. Add raw files to data/raw/ first.")
        sys.exit(1)

    train_bpe_tokenizer(files=files, vocab_size=args.vocab_size, save_dir=args.save_dir)


def command_prepare_data(args):
    """Clean, tokenize, and split dataset into train and validation tensors."""
    from training.dataset import prepare_and_tokenize_data

    prepare_and_tokenize_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        tokenizer_dir=args.tokenizer_dir,
        train_ratio=args.train_ratio,
    )


def command_train(args):
    """Train the model from random initialization or specified parameters."""
    from config.model_config import get_model_config
    from config.training_config import get_training_config
    from tokenizer.tokenizer_utils import OmegaTokenizer
    from training.trainer import OmegaTrainer

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
    trainer.train(resume=args.resume)


def command_resume(args):
    """Resume training from latest saved checkpoint."""
    from config.model_config import get_model_config
    from config.training_config import get_training_config
    from tokenizer.tokenizer_utils import OmegaTokenizer
    from training.trainer import OmegaTrainer

    model_cfg = get_model_config(profile=args.profile)
    overrides = {"device": args.device}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps

    train_cfg = get_training_config(**overrides)

    tokenizer = OmegaTokenizer.load(train_cfg.tokenizer_dir)
    model_cfg.vocab_size = tokenizer.get_vocab_size()

    trainer = OmegaTrainer(model_config=model_cfg, training_config=train_cfg, tokenizer=tokenizer)
    trainer.train(resume=True)


def command_generate(args):
    """Generate text from trained model or launch interactive console."""
    from generation.generate import TextGenerator
    from generation.interactive_chat import run_interactive_session

    if args.interactive or args.prompt is None:
        run_interactive_session(checkpoint_path=args.checkpoint, device=args.device)
    else:
        gen = TextGenerator(checkpoint_path=args.checkpoint, device=args.device)
        res = gen.generate(
            prompt=args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            save_path=args.save_path,
        )
        print("\n" + "=" * 50)
        print("OMEGA GPT-1 GENERATED OUTPUT:")
        print("=" * 50)
        print(res["generated_text"])
        print("-" * 50)
        print(f"Speed: {res['tokens_per_second']:.1f} tokens/sec ({res['tokens_generated']} tokens)")


def command_evaluate(args):
    """Calculate validation perplexity."""
    from evaluation.perplexity import calculate_dataset_perplexity

    calculate_dataset_perplexity(
        checkpoint_path=args.checkpoint,
        tokens_path=args.tokens,
        device=args.device,
    )


def command_test(args):
    """Run full automated test suite."""
    print("=" * 60)
    print("RUNNING OMEGA GPT-1 AUTOMATED TEST SUITE")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Omega GPT-1 Command Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # info
    p_info = subparsers.add_parser("info", help="Display model architecture, parameters, and system diagnostics")
    p_info.add_argument("--profile", type=str, default="small", choices=["tiny", "small", "medium", "custom"])

    # setup
    subparsers.add_parser("setup", help="Verify environment and create directory structure")

    # train-tokenizer
    p_tok = subparsers.add_parser("train-tokenizer", help="Train BPE tokenizer from scratch")
    p_tok.add_argument("--data_dir", type=str, default="data/raw")
    p_tok.add_argument("--vocab_size", type=int, default=8000)
    p_tok.add_argument("--save_dir", type=str, default="tokenizer/saved")

    # prepare-data
    p_data = subparsers.add_parser("prepare-data", help="Clean, tokenize, and split dataset")
    p_data.add_argument("--raw_dir", type=str, default="data/raw")
    p_data.add_argument("--processed_dir", type=str, default="data/processed")
    p_data.add_argument("--tokenizer_dir", type=str, default="tokenizer/saved")
    p_data.add_argument("--train_ratio", type=float, default=0.9)

    # train
    p_train = subparsers.add_parser("train", help="Train model from scratch")
    p_train.add_argument("--profile", type=str, default="small", choices=["tiny", "small", "medium", "custom"])
    p_train.add_argument("--max_steps", type=int, default=None)
    p_train.add_argument("--batch_size", type=int, default=None)
    p_train.add_argument("--learning_rate", type=float, default=None)
    p_train.add_argument("--device", type=str, default="auto")
    p_train.add_argument("--resume", action="store_true")

    # resume
    p_res = subparsers.add_parser("resume", help="Resume interrupted training from latest checkpoint")
    p_res.add_argument("--profile", type=str, default="small")
    p_res.add_argument("--max_steps", type=int, default=None)
    p_res.add_argument("--device", type=str, default="auto")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate text from model")
    p_gen.add_argument("--prompt", type=str, default=None)
    p_gen.add_argument("--checkpoint", type=str, default=None)
    p_gen.add_argument("--tokens", type=int, default=120)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top_k", type=int, default=40)
    p_gen.add_argument("--top_p", type=float, default=0.9)
    p_gen.add_argument("--save_path", type=str, default=None)
    p_gen.add_argument("--interactive", action="store_true")
    p_gen.add_argument("--device", type=str, default="auto")

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Compute validation perplexity")
    p_eval.add_argument("--checkpoint", type=str, default="outputs/checkpoints/omega_gpt1_best.pth")
    p_eval.add_argument("--tokens", type=str, default="data/processed/val_tokens.pt")
    p_eval.add_argument("--device", type=str, default="auto")

    # test
    subparsers.add_parser("test", help="Run automated test suite")

    args = parser.parse_args()

    commands = {
        "info": command_info,
        "setup": command_setup,
        "train-tokenizer": command_train_tokenizer,
        "prepare-data": command_prepare_data,
        "train": command_train,
        "resume": command_resume,
        "generate": command_generate,
        "evaluate": command_evaluate,
        "test": command_test,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
