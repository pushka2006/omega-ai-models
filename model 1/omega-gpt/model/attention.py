"""Causal Multi-Head Self-Attention for Omega GPT-1."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention with autoregressive lower-triangular masking."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        # Q, K, V combined projection for computational efficiency
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer: upper triangular values set to -inf
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:

        x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor of shape (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()

        # Calculate query, key, values
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch 2.0+ Scaled Dot-Product Attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            # is_causal enforces triangular autoregressive mask
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Explicit scaled dot-product attention calculation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Concatenate heads: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear output projection
        return self.resid_dropout(self.c_proj(y))

