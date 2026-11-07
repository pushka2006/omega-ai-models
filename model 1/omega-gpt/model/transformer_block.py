"""Transformer Block for Omega GPT-1 supporting GELU and SwiGLU activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import ModelConfig
from .attention import CausalSelfAttention


class FeedForward(nn.Module):
    """Feed-Forward Network supporting standard GELU and modern SwiGLU activations."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.activation_type = getattr(config, "activation", "gelu").lower()

        if self.activation_type == "swiglu":
            # SwiGLU uses gate and value projections
            self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.w_val = nn.Linear(config.d_model, config.d_ff, bias=False)
            self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        else:
            # Standard GELU / ReLU
            self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=False)
            if self.activation_type == "relu":
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()
            self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            # SwiGLU(x) = (w_gate(x) * SiLU(w_val(x))) @ c_proj
            gate = self.w_gate(x)
            val = F.silu(self.w_val(x))
            x = self.c_proj(gate * val)
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)

        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Single decoder Transformer block with Pre-LayerNorm and residual connections."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention with residual
        x = x + self.attn(self.ln_1(x))
        # Pre-LN Feed-Forward with residual
        x = x + self.mlp(self.ln_2(x))
        return x
