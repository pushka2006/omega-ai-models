"""Embedding layers for Omega GPT-1."""

import torch
import torch.nn as nn
from config.model_config import ModelConfig


class OmegaEmbeddings(nn.Module):
    """Combines learned token embeddings and learned positional embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor = None) -> torch.Tensor:
        """Args:

        input_ids: LongTensor of shape (batch_size, seq_len)
        position_ids: Optional LongTensor of shape (batch_size, seq_len)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds model context length ({self.config.context_length})"
            )

        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)  # Shape: (1, seq_len)

        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(position_ids)

        embeddings = tok_emb + pos_emb
        return self.dropout(embeddings)

