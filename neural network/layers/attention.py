"""OMEGA NEURAL NETWORK ENGINE: Attention and Transformer Layers."""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention supporting causal and bidirectional attention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.is_causal = is_causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, q: torch.Tensor, k: Optional[torch.Tensor] = None, v: Optional[torch.Tensor] = None) -> torch.Tensor:
        k = k if k is not None else q
        v = v if v is not None else q
        B, T_q, _ = q.size()
        _, T_k, _ = k.size()

        # Projections & head reshape: (B, T, d_model) -> (B, num_heads, T, head_dim)
        q_h = self.q_proj(q).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = self.k_proj(k).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = self.v_proj(v).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q_h, k_h, v_h, is_causal=self.is_causal, dropout_p=drop_p)

        # Concatenate heads & project
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)


class ResidualConnection(nn.Module):
    """Residual Skip Connection wrapper: y = x + sublayer(x)."""

    def __init__(self, sublayer: nn.Module):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x + self.sublayer(x, *args, **kwargs)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int = 1024, dropout: float = 0.1, is_causal: bool = False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout, is_causal=is_causal)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.ff(self.ln_2(x))
        return x
