"""Production Byte Pair Encoding (BPE) Tokenizer for Omega GPT-1.

Supports full UTF-8/Unicode, punctuation, subwords, and special tokens:
[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3.
"""

import json
import os
from typing import List, Optional, Union

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class OmegaTokenizer:
    """Production BPE Tokenizer wrapper for Omega GPT-1."""

    def __init__(self, tokenizer_backend=None):
        self._backend = tokenizer_backend
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.bos_token = BOS_TOKEN
        self.eos_token = EOS_TOKEN

        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    @classmethod
    def load(cls, save_dir: str = "tokenizer/saved") -> "OmegaTokenizer":
        """Load trained tokenizer and metadata from save_dir."""
        # Check both relative and absolute paths
        candidates = [
            save_dir,
            os.path.join(os.path.dirname(__file__), "saved"),
            os.path.join(os.path.dirname(__file__), "..", "tokenizer", "saved"),
        ]

        target_dir = None
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "tokenizer.json")) or os.path.exists(os.path.join(cand, "vocab.json")):
                target_dir = cand
                break

        if target_dir is None:
            raise FileNotFoundError(
                f"Tokenizer files not found in '{save_dir}'. Run 'python main.py train-tokenizer' first."
            )

        json_path = os.path.join(target_dir, "tokenizer.json")
        if os.path.exists(json_path):
            try:
                from tokenizers import Tokenizer
                hf_tok = Tokenizer.from_file(json_path)
                instance = cls(tokenizer_backend=hf_tok)
                # Map special token IDs
                p_id = hf_tok.token_to_id(PAD_TOKEN)
                u_id = hf_tok.token_to_id(UNK_TOKEN)
                b_id = hf_tok.token_to_id(BOS_TOKEN)
                e_id = hf_tok.token_to_id(EOS_TOKEN)

                instance.pad_id = p_id if p_id is not None else PAD_ID
                instance.unk_id = u_id if u_id is not None else UNK_ID
                instance.bos_id = b_id if b_id is not None else BOS_ID
                instance.eos_id = e_id if e_id is not None else EOS_ID
                return instance
            except ImportError:
                pass

        # Fallback vocab.json loader
        vocab_path = os.path.join(target_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            fallback = StandaloneBPETokenizer(vocab)
            return cls(tokenizer_backend=fallback)

        raise FileNotFoundError(f"Neither tokenizer.json nor vocab.json found in {target_dir}")

    def save(self, save_dir: str):
        """Save tokenizer configuration and human-readable metadata."""
        os.makedirs(save_dir, exist_ok=True)
        if self._backend is not None and hasattr(self._backend, "save"):
            self._backend.save(os.path.join(save_dir, "tokenizer.json"))
            vocab = self.get_vocab()
            with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)

        # Save metadata for compatibility checking
        metadata = {
            "vocab_size": self.get_vocab_size(),
            "special_tokens": {
                PAD_TOKEN: self.pad_id,
                UNK_TOKEN: self.unk_id,
                BOS_TOKEN: self.bos_id,
                EOS_TOKEN: self.eos_id,
            },
        }
        with open(os.path.join(save_dir, "tokenizer_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode UTF-8 string into token IDs."""
        if not text:
            return []

        if self._backend is not None and hasattr(self._backend, "encode"):
            encoded = self._backend.encode(text)
            ids = list(encoded.ids)
        elif self._backend is not None and hasattr(self._backend, "encode_text"):
            ids = self._backend.encode_text(text)
        else:
            ids = [ord(c) for c in text]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]

        return ids

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into string with validity bounds checking."""
        if not tokens:
            return ""

        vocab_size = self.get_vocab_size()
        # Filter invalid out-of-range token IDs
        valid_tokens = [t for t in tokens if 0 <= t < vocab_size]

        if self._backend is not None and hasattr(self._backend, "decode"):
            try:
                return self._backend.decode(valid_tokens, skip_special_tokens=skip_special_tokens)
            except Exception:
                pass

        # Fallback decode
        filtered = valid_tokens
        if skip_special_tokens:
            special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
            filtered = [t for t in valid_tokens if t not in special_ids]

        vocab = self.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        return "".join(inv_vocab.get(t, "") for t in filtered)

    def get_vocab_size(self) -> int:
        """Return total vocabulary size."""
        if self._backend is not None and hasattr(self._backend, "get_vocab_size"):
            return self._backend.get_vocab_size()
        elif self._backend is not None and hasattr(self._backend, "vocab"):
            return len(self._backend.vocab)
        return 8000

    def get_vocab(self) -> dict:
        if self._backend is not None and hasattr(self._backend, "get_vocab"):
            return self._backend.get_vocab()
        elif self._backend is not None and hasattr(self._backend, "vocab"):
            return self._backend.vocab
        return {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}


class StandaloneBPETokenizer:
    """Standalone BPE Tokenizer fallback for lightweight or zero-dependency runs."""

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode_text(self, text: str) -> List[int]:
        return [self.vocab.get(c, UNK_ID) for c in text]

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for t in tokens:
            if skip_special_tokens and t in (PAD_ID, UNK_ID, BOS_ID, EOS_ID):
                continue
            chars.append(self.inv_vocab.get(t, ""))
        return "".join(chars)

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict:
        return self.vocab
