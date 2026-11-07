"""Omega GPT-1: Core Transformer Language Model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_config import ModelConfig
from .embeddings import OmegaEmbeddings
from .model_utils import count_parameters, format_parameter_count, init_weights
from .transformer_block import TransformerBlock


class OmegaGPT(nn.Module):
    """Omega GPT-1: Complete decoder-only Transformer language model trained from scratch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Embeddings: Token + Positional
        self.embeddings = OmegaEmbeddings(config)

        # Transformer blocks stack
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])

        # Final LayerNorm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language Modeling Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embeddings and LM head
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.token_embeddings.weight

        # Initialize all weights randomly
        self.apply(lambda module: init_weights(module, config.initializer_range, config.num_layers))

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Args:

        input_ids: LongTensor of shape (batch_size, seq_len)
        targets: Optional LongTensor of shape (batch_size, seq_len)
        Returns:
            (logits, loss): logits shape (batch_size, seq_len, vocab_size)
        """
        B, T = input_ids.size()
        if T > self.config.context_length:
            raise ValueError(
                f"Input sequence length {T} exceeds maximum context length {self.config.context_length}"
            )

        # Embed tokens and positions
        x = self.embeddings(input_ids)

        # Pass sequentially through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.ln_f(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # Flatten B and T dimensions for cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)

        return logits, loss

    def configure_optimizers(
        self,
        weight_decay: float = 0.01,
        learning_rate: float = 3e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cpu",
    ) -> torch.optim.Optimizer:
        """Configures AdamW with weight decay applied exclusively to 2D matrix parameters

        (weights) and excluded from 1D parameters (biases, layernorms).
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Use fused AdamW if CUDA is available and supported
        use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
        extra_args = {"fused": True} if use_fused else {}

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_id: int = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with temperature, top-k, top-p, and repetition penalty.

        Args:
            idx: Tensor of shape (batch_size, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (1.0 = normal, <1.0 = conservative, >1.0 = creative)
            top_k: Restrict sampling to top K highest probability tokens
            top_p: Nucleus sampling cumulative probability threshold
            repetition_penalty: Penalty factor applied to previously seen token logits
            eos_id: End of sequence token ID to terminate generation early
        """
        for _ in range(max_new_tokens):
            # Crop to maximum context window if sequence length exceeds limit
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # Forward pass to get logits at the final position
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen_tokens = set(idx[b].tolist())
                    for token in seen_tokens:
                        if logits[b, token] < 0:
                            logits[b, token] *= repetition_penalty
                        else:
                            logits[b, token] /= repetition_penalty

            # Apply temperature
            if temperature > 0.0:
                logits = logits / temperature
            else:
                # Greedy argmax if temperature is zero
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                if eos_id is not None and (idx_next == eos_id).all():
                    break
                continue

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            # Sample from the filtered probability distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)

            # Check if all batches produced EOS
            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx

    def print_summary(self):
        """Display model architecture summary and parameter counts."""
        total, trainable = count_parameters(self)
        print("=" * 60)
        print("OMEGA GPT-1 ARCHITECTURE SUMMARY")
        print("=" * 60)
        print(f"Layers (Blocks)       : {self.config.num_layers}")
        print(f"Attention Heads       : {self.config.num_heads}")
        print(f"Embedding Dimension   : {self.config.d_model}")
        print(f"Feed-Forward Dim      : {self.config.d_ff}")
        print(f"Context Length        : {self.config.context_length}")
        print(f"Vocabulary Size       : {self.config.vocab_size}")
        print(f"Weight Tying          : {self.config.tie_word_embeddings}")
        print(f"Dropout Rate          : {self.config.dropout}")
        print("-" * 60)
        print(f"Total Parameters      : {total:,} ({format_parameter_count(total)})")
        print(f"Trainable Parameters  : {trainable:,} ({format_parameter_count(trainable)})")
        print("=" * 60)

