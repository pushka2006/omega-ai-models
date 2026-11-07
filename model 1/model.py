"""Omega GPT-1: Modern Decoder-Only Transformer Language Model (Model 1).

Implements Pre-LayerNorm architecture, PyTorch Scaled Dot-Product Attention (SDPA),
GELU activation, weight tying, and autoregressive generation with nucleus (top-p),
top-k, temperature, and repetition penalty controls.
Directly compatible with trained checkpoints in checkpoints/ and omega-gpt/.
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from config import (
        activation as default_activation,
        block_size as default_block_size,
        d_ff as default_d_ff,
        device as default_device,
        dropout as default_dropout,
        n_embd as default_n_embd,
        n_head as default_n_head,
        n_layer as default_n_layer,
        tie_weights as default_tie_weights,
        vocab_size as default_vocab_size,
    )
except ImportError:
    default_vocab_size = 1200
    default_block_size = 128
    default_n_embd = 256
    default_n_head = 4
    default_n_layer = 4
    default_d_ff = 1024
    default_dropout = 0.1
    default_activation = "gelu"
    default_tie_weights = True
    default_device = "cuda" if torch.cuda.is_available() else "cpu"


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention powered by PyTorch SDPA."""

    def __init__(self, d_model: int, num_heads: int, context_length: int = 128, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout

        # Fused linear projection for Query, Key, and Value
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Project and separate Q, K, V: shape (B, T, 3 * C) -> 3 x (B, num_heads, T, head_dim)
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # PyTorch SDPA
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=self.dropout_p if self.training else 0.0
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(out))


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU/SwiGLU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.activation_type = activation.lower()

        if self.activation_type == "swiglu":
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w_val = nn.Linear(d_model, d_ff, bias=False)
            self.c_proj = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.c_fc = nn.Linear(d_model, d_ff, bias=False)
            if self.activation_type == "relu":
                self.act = nn.ReLU()
            else:
                self.act = nn.GELU()
            self.c_proj = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            gate = self.w_gate(x)
            val = F.silu(self.w_val(x))
            x = self.c_proj(gate * val)
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with residual connections."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int = 128,
        d_ff: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, context_length=context_length, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout=dropout, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class OmegaEmbeddings(nn.Module):
    """Token and learned positional embeddings."""

    def __init__(self, vocab_size: int, context_length: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.size()
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        tok_emb = self.token_embeddings(input_ids)
        pos_emb = self.position_embeddings(positions)
        return self.dropout(tok_emb + pos_emb)


class GPTLanguageModel(nn.Module):
    """Production Decoder-Only Transformer Language Model (Omega GPT-1)."""

    def __init__(
        self,
        vocab_size: int = default_vocab_size,
        block_size: int = default_block_size,
        n_embd: int = default_n_embd,
        n_head: int = default_n_head,
        n_layer: int = default_n_layer,
        d_ff: Optional[int] = None,
        dropout: float = default_dropout,
        activation: str = default_activation,
        tie_weights: bool = default_tie_weights,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.context_length = block_size
        self.n_embd = n_embd
        self.d_model = n_embd
        self.n_head = n_head
        self.num_heads = n_head
        self.n_layer = n_layer
        self.num_layers = n_layer
        self.d_ff = d_ff if d_ff is not None else 4 * n_embd
        self.dropout_rate = dropout
        self.activation = activation
        self.tie_weights = tie_weights

        # Embeddings
        self.embeddings = OmegaEmbeddings(self.vocab_size, self.block_size, self.n_embd, dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.n_embd,
                self.n_head,
                context_length=self.block_size,
                d_ff=self.d_ff,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(self.n_layer)
        ])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(self.n_embd)

        # Language Modeling Head
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

        # Weight tying
        if self.tie_weights:
            self.lm_head.weight = self.embeddings.token_embeddings.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        if T > self.block_size:
            idx = idx[:, -self.block_size :]
            T = idx.size(1)

        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            if targets.size(1) > self.block_size:
                targets = targets[:, -self.block_size :]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = 3,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k, top-p, temperature, and repetition penalty."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size :]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Next-token logits: (B, vocab_size)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen_tokens = set(idx[b].tolist())
                    for token_id in seen_tokens:
                        if token_id < logits.size(-1):
                            if logits[b, token_id] > 0:
                                logits[b, token_id] /= repetition_penalty
                            else:
                                logits[b, token_id] *= repetition_penalty

            # Temperature scaling
            if temperature > 0.0:
                logits = logits / max(temperature, 1e-6)

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    k_val = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, k_val, dim=-1)[0][:, -1, None]
                    logits[logits < threshold] = float("-inf")

                # Top-P (nucleus) filtering
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    for b in range(logits.size(0)):
                        remove_mask = sorted_indices_to_remove[b]
                        indices_to_remove = sorted_indices[b][remove_mask]
                        logits[b, indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx

    def count_parameters(self) -> Tuple[int, int]:
        """Return (total_parameters, trainable_parameters)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def configure_optimizers(
        self,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
    ) -> torch.optim.AdamW:
        """Configure AdamW with decoupled weight decay for 2D weight matrices only."""
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


# Aliases for unified naming
OmegaGPT = GPTLanguageModel
Head = CausalSelfAttention
MultiHeadAttention = CausalSelfAttention
Block = TransformerBlock
