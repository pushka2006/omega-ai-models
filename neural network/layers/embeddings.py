"""OMEGA NEURAL NETWORK ENGINE: Embedding Layers."""

import math
from typing import Optional
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Vocabulary Token Embedding Layer."""

    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class LearnedPositionalEmbedding(nn.Module):
    """Learned absolute positional embedding layer."""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        return self.embedding(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal positional encoding as formulated in Attention Is All You Need."""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, embedding_dim)
        return x + self.pe[:, : x.size(1), :]
