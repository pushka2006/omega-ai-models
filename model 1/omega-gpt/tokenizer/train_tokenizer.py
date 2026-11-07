"""Train a Byte Pair Encoding (BPE) Tokenizer from scratch for Omega GPT-1."""

import argparse
import glob
import os
from typing import List, Optional

from .tokenizer_utils import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKENS,
    UNK_TOKEN,
    OmegaTokenizer,
)


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int = 8000,
    min_frequency: int = 2,
    save_dir: str = "tokenizer/saved",
) -> OmegaTokenizer:
    """Train a custom BPE tokenizer from scratch on local text files."""
    if not files:
        raise ValueError("No files provided for tokenizer training.")

    existing_files = [f for f in files if os.path.exists(f)]
    if not existing_files:
        raise FileNotFoundError(f"None of the specified files exist: {files}")

    print(f"[Tokenizer] Training BPE tokenizer on {len(existing_files)} file(s)...")
    print(f"[Tokenizer] Target vocabulary size: {vocab_size}, min frequency: {min_frequency}")

    try:
        from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.normalizer = normalizers.Sequence([
            normalizers.NFC(),
        ])
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )

        tokenizer.train(files=existing_files, trainer=trainer)

        wrapper = OmegaTokenizer(tokenizer_backend=tokenizer)
        wrapper.save(save_dir)

        actual_vocab_size = wrapper.get_vocab_size()
        print(f"[Tokenizer] Training complete. Vocabulary size: {actual_vocab_size}")
        print(f"[Tokenizer] Tokenizer saved successfully to '{save_dir}'.")

        # Sanity check
        sample_text = "Omega GPT-1 is an original Transformer language model trained from scratch."
        encoded = wrapper.encode(sample_text)
        decoded = wrapper.decode(encoded)
        print(f"[Tokenizer] Sanity Check: '{sample_text}'")
        print(f"[Tokenizer] Encoded tokens ({len(encoded)}): {encoded[:10]}...")
        print(f"[Tokenizer] Decoded text: '{decoded}'")

        return wrapper

    except ImportError:
        print("[Tokenizer] Hugging Face tokenizers not available. Training fallback character/subword tokenizer.")
        # Minimal fallback
        import json
        char_set = set()
        for fpath in existing_files:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                char_set.update(f.read())

        vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
        for c in sorted(char_set):
            if c not in vocab and len(vocab) < vocab_size:
                vocab[c] = len(vocab)

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        return OmegaTokenizer.load(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Omega GPT-1 BPE Tokenizer")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Path to raw text files")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--save_dir", type=str, default="tokenizer/saved", help="Directory to save tokenizer")
    args = parser.parse_args()

    file_patterns = [
        os.path.join(args.data_dir, "*.txt"),
        os.path.join(args.data_dir, "**", "*.txt"),
    ]
    matched_files = []
    for pattern in file_patterns:
        matched_files.extend(glob.glob(pattern, recursive=True))
    matched_files = sorted(list(set(matched_files)))

    if not matched_files:
        print(f"Error: No .txt files found in {args.data_dir}")
        exit(1)

    train_bpe_tokenizer(matched_files, vocab_size=args.vocab_size, save_dir=args.save_dir)

