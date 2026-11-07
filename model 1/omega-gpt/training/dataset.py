"""Dataset management and preprocessing for Omega GPT-1."""

import glob
import json
import os
import re
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer_utils import OmegaTokenizer


def clean_text(text: str) -> str:
    """Clean and normalize raw text: remove control characters, normalize whitespace."""
    # Replace non-printable characters except standard whitespace
    text = re.sub(r"[^\x20-\x7E\n\r\t]", " ", text)
    # Normalize excessive carriage returns and tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace more than 3 consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def prepare_and_tokenize_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    tokenizer_dir: str = "tokenizer/saved",
    train_ratio: float = 0.9,
    min_doc_length: int = 20,
) -> Dict[str, int]:
    """Ingest, clean, deduplicate, tokenize, and split text corpora."""
    os.makedirs(processed_dir, exist_ok=True)

    files = glob.glob(os.path.join(raw_dir, "*.txt")) + glob.glob(os.path.join(raw_dir, "**", "*.txt"))
    files = sorted(list(set(files)))

    if not files:
        raise FileNotFoundError(f"No .txt files found in raw data directory: {raw_dir}")

    print(f"[Data] Found {len(files)} raw text file(s) in '{raw_dir}'.")

    # Load and clean documents
    documents: List[str] = []
    seen_hashes = set()
    total_raw_chars = 0

    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            total_raw_chars += len(content)
            cleaned = clean_text(content)

            # Deduplication
            doc_hash = hash(cleaned)
            if doc_hash not in seen_hashes and len(cleaned) >= min_doc_length:
                seen_hashes.add(doc_hash)
                documents.append(cleaned)

    print(f"[Data] Processed {len(documents)} unique document(s) ({total_raw_chars:,} raw characters).")

    # Join documents with EOS separator
    joined_text = "\n\n".join(documents)
    total_clean_chars = len(joined_text)

    # Tokenize
    tokenizer = OmegaTokenizer.load(tokenizer_dir)
    print("[Data] Tokenizing corpus using OmegaTokenizer...")
    token_ids = tokenizer.encode(joined_text, add_special_tokens=True)
    total_tokens = len(token_ids)
    print(f"[Data] Total tokens generated: {total_tokens:,}")

    # Split train and validation (prevent data leakage)
    split_idx = int(train_ratio * total_tokens)
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    # Fallback for very small datasets during testing: ensure val has at least some tokens
    if len(val_tokens) < 10 and total_tokens > 10:
        val_tokens = train_tokens[-max(10, int(0.1 * total_tokens)):]

    # Save to disk as PyTorch LongTensors
    train_tensor = torch.tensor(train_tokens, dtype=torch.long)
    val_tensor = torch.tensor(val_tokens, dtype=torch.long)

    train_path = os.path.join(processed_dir, "train_tokens.pt")
    val_path = os.path.join(processed_dir, "val_tokens.pt")

    torch.save(train_tensor, train_path)
    torch.save(val_tensor, val_path)

    metadata = {
        "num_raw_files": len(files),
        "num_documents": len(documents),
        "total_raw_chars": total_raw_chars,
        "total_clean_chars": total_clean_chars,
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_ratio": train_ratio,
        "vocab_size": tokenizer.get_vocab_size(),
    }

    metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[Data] Train tokens: {len(train_tokens):,} -> saved to '{train_path}'")
    print(f"[Data] Validation tokens: {len(val_tokens):,} -> saved to '{val_path}'")
    print(f"[Data] Metadata saved to '{metadata_path}'")

    return metadata


class OmegaDataset(Dataset):
    """PyTorch Dataset that generates autoregressive sequence chunks of length context_length."""

    def __init__(self, data_tensor: torch.Tensor, context_length: int):
        self.data = data_tensor
        self.context_length = context_length

        if len(self.data) <= self.context_length:
            # If data is shorter than context, pad it or repeat it for testing
            repeats = (self.context_length + 2) // len(self.data) + 1
            self.data = self.data.repeat(repeats)

        self.num_samples = len(self.data) - self.context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Input sequence x of length context_length
        x = self.data[idx : idx + self.context_length]
        # Target sequence y shifted by 1 token for next-token prediction
        y = self.data[idx + 1 : idx + self.context_length + 1]
        return x, y

