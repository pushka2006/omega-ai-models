"""OMEGA NEXUS MODEL: Multimodal Cross-Attention & Latent Fusion Layer.

Fuses heterogeneous representations (continuous features, neural embeddings,
and language tokens) into a unified semantic manifold.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalProjector(nn.Module):
    """Projects continuous tabular or visual features into the Nexus latent space."""

    def __init__(self, input_dim: int, latent_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism between language tokens and auxiliary neural features."""

    def __init__(self, latent_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """Cross-attend query (e.g. text representations) over key_value (auxiliary neural features)."""
        B = query.size(0)

        # Pre-LN
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        # Projections & reshape
        Q = self.q_proj(q_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_norm).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(Q, K, V, dropout_p=self.dropout.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, -1, self.latent_dim)
        proj_out = self.out_proj(attn_out)

        # Gated residual connection
        return query + torch.tanh(self.gate) * proj_out


class NexusFusionNetwork(nn.Module):
    """End-to-end multimodal fusion network combining tabular, neural, and text modalities."""

    def __init__(
        self,
        text_dim: int = 256,
        feature_dim: int = 32,
        latent_dim: int = 256,
        num_classes: int = 5,
        num_heads: int = 4,
    ):
        super().__init__()
        self.feature_projector = MultimodalProjector(feature_dim, latent_dim)
        self.text_projector = MultimodalProjector(text_dim, latent_dim)
        self.cross_attn = CrossAttentionFusion(latent_dim, num_heads)

        self.classifier_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Linear(latent_dim // 2, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, feature_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns fused representations and downstream classification logits."""
        # Ensure 3D tensors: (Batch, Seq_Len, Dim)
        if text_emb.dim() == 2:
            text_emb = text_emb.unsqueeze(1)
        if feature_tensor.dim() == 2:
            feature_tensor = feature_tensor.unsqueeze(1)

        t_proj = self.text_projector(text_emb)
        f_proj = self.feature_projector(feature_tensor)

        fused = self.cross_attn(query=t_proj, key_value=f_proj)
        pooled = fused.mean(dim=1)
        logits = self.classifier_head(pooled)

        return fused, logits
