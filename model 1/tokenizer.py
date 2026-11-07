"""Unified Tokenizer Interface for Omega GPT-1 (Model 1).

Implements Byte Pair Encoding (BPE) with full UTF-8 & Unicode support,
special tokens handling ([PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3), and maintains
backward compatibility with CharacterTokenizer.
"""

import json
import os
from typing import List, Optional, Union
import torch

# Special Tokens Configuration
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
BOS_TOKEN = "[BOS]"
EOS_TOKEN = "[EOS]"

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class StandaloneBPETokenizer:
    """Fallback subword tokenizer when Hugging Face tokenizers is not present."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = []
        i = 0
        n = len(text)
        sorted_keys = sorted(self.vocab.keys(), key=lambda s: len(s), reverse=True)
        while i < n:
            matched = False
            for token_str in sorted_keys:
                if text.startswith(token_str, i):
                    tokens.append(self.vocab[token_str])
                    i += len(token_str)
                    matched = True
                    break
            if not matched:
                tokens.append(UNK_ID)
                i += 1
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        res = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tok = self.id_to_token[tid]
                if tok not in SPECIAL_TOKENS:
                    res.append(tok)
            else:
                res.append(UNK_TOKEN)
        return "".join(res)


class OmegaTokenizer:
    """Production BPE Tokenizer for Omega GPT-1."""

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
    def load(cls, save_dir: Optional[str] = None) -> "OmegaTokenizer":
        """Load trained tokenizer from save_dir or standard candidate locations."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        candidates = []
        if save_dir:
            candidates.append(save_dir)
        candidates.extend([
            os.path.join(base_dir, "tokenizer_saved"),
            os.path.join(base_dir, "omega-gpt", "tokenizer", "saved"),
            os.path.join(base_dir, "saved"),
            "tokenizer_saved",
            "omega-gpt/tokenizer/saved",
        ])

        target_dir = None
        for cand in candidates:
            if os.path.exists(os.path.join(cand, "tokenizer.json")) or os.path.exists(os.path.join(cand, "vocab.json")):
                target_dir = cand
                break

        if target_dir is None:
            raise FileNotFoundError(
                f"Tokenizer files not found in candidate paths. Run 'python tokenizer.py' to train it first."
            )

        json_path = os.path.join(target_dir, "tokenizer.json")
        if os.path.exists(json_path):
            try:
                from tokenizers import Tokenizer
                hf_tok = Tokenizer.from_file(json_path)
                instance = cls(tokenizer_backend=hf_tok)
                p_id = hf_tok.token_to_id(PAD_TOKEN)
                u_id = hf_tok.token_to_id(UNK_TOKEN)
                b_id = hf_tok.token_to_id(BOS_TOKEN)
                e_id = hf_tok.token_to_id(EOS_TOKEN)

                instance.pad_id = p_id if p_id is not None else PAD_ID
                instance.unk_id = u_id if u_id is not None else UNK_ID
                instance.bos_id = b_id if b_id is not None else BOS_ID
                instance.eos_id = e_id if e_id is not None else EOS_ID
                return instance
            except Exception:
                pass

        # Fallback to vocab.json
        vocab_path = os.path.join(target_dir, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
            fallback = StandaloneBPETokenizer(vocab)
            return cls(tokenizer_backend=fallback)

        raise FileNotFoundError(f"Neither tokenizer.json nor vocab.json could be parsed from {target_dir}")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to token ID sequence."""
        if not text:
            return []
        if self._backend is None:
            return [ord(c) for c in text]

        if hasattr(self._backend, "encode"):
            result = self._backend.encode(text)
            ids = result.ids if hasattr(result, "ids") else result
        else:
            ids = [ord(c) for c in text]

        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token ID sequence back to string."""
        if not token_ids:
            return ""

        valid_ids = [t for t in token_ids if isinstance(t, int) and 0 <= t < self.get_vocab_size()]

        if hasattr(self._backend, "decode"):
            if hasattr(self._backend, "token_to_id"):
                return self._backend.decode(valid_ids, skip_special_tokens=skip_special_tokens)
            return self._backend.decode(valid_ids)

        return "".join(chr(t) for t in valid_ids if t < 128)

    def encode_to_tensor(self, text: str, device: str = "cpu", add_special_tokens: bool = False) -> torch.Tensor:
        """Encode text directly to a 2D LongTensor of shape (1, seq_len)."""
        tokens = self.encode(text, add_special_tokens=add_special_tokens)
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def get_vocab_size(self) -> int:
        """Return the total vocabulary size."""
        if self._backend is not None and hasattr(self._backend, "get_vocab_size"):
            return self._backend.get_vocab_size()
        if self._backend is not None and hasattr(self._backend, "vocab"):
            return len(self._backend.vocab)
        return 1200

    def get_vocab(self) -> dict[str, int]:
        """Return token-to-id mapping dictionary."""
        if self._backend is not None and hasattr(self._backend, "get_vocab"):
            return self._backend.get_vocab()
        if self._backend is not None and hasattr(self._backend, "vocab"):
            return self._backend.vocab
        return {SPECIAL_TOKENS[i]: i for i in range(len(SPECIAL_TOKENS))}

    def save(self, save_dir: str):
        """Save tokenizer configuration and human-readable metadata."""
        os.makedirs(save_dir, exist_ok=True)
        if self._backend is not None and hasattr(self._backend, "save"):
            self._backend.save(os.path.join(save_dir, "tokenizer.json"))
            vocab = self.get_vocab()
            with open(os.path.join(save_dir, "vocab.json"), "w", encoding="utf-8") as f:
                json.dump(vocab, f, ensure_ascii=False, indent=2)

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


class CharacterTokenizer:
    """ASCII character-level tokenizer (maintained for backwards compatibility)."""

    def __init__(self, vocab_size: int = 128):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text if ord(c) < self.vocab_size]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens if 0 <= t < self.vocab_size)

    def encode_to_tensor(self, text: str, device: str = "cpu") -> torch.Tensor:
        tokens = self.encode(text)
        return torch.tensor([tokens], dtype=torch.long, device=device)

    def get_vocab_size(self) -> int:
        return self.vocab_size


def get_tokenizer(path: Optional[str] = None) -> Union[OmegaTokenizer, CharacterTokenizer]:
    """Factory function: attempts to load the trained BPE tokenizer first, fallback to CharacterTokenizer."""
    try:
        return OmegaTokenizer.load(path)
    except Exception:
        return CharacterTokenizer()


def train_bpe_tokenizer(
    corpus_files: list[str],
    vocab_size: int = 1200,
    save_dir: str = "tokenizer_saved",
) -> OmegaTokenizer:
    """Train a Byte Pair Encoding (BPE) tokenizer from corpus text files."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    valid_files = [f for f in corpus_files if os.path.exists(f) and os.path.getsize(f) > 0]
    if not valid_files:
        raise ValueError(f"No valid corpus files provided from {corpus_files}")

    tokenizer.train(files=valid_files, trainer=trainer)
    wrapper = OmegaTokenizer(tokenizer_backend=tokenizer)
    wrapper.save(save_dir)
    return wrapper


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(base, "data", "training.txt")
    save_target = os.path.join(base, "tokenizer_saved")

    print(f"Training BPE tokenizer on {data_file} (vocab_size=1200)...")
    tok = train_bpe_tokenizer([data_file], vocab_size=1200, save_dir=save_target)
    print(f"Tokenizer trained successfully! Vocab size: {tok.get_vocab_size()}")
    sample_text = "Transformers utilize self-attention mechanisms."
    encoded = tok.encode(sample_text)
    decoded = tok.decode(encoded)
    print(f"Sample Text : '{sample_text}'")
    print(f"Token IDs   : {encoded}")
    print(f"Decoded     : '{decoded}'")
