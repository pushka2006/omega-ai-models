"""Sampling algorithms for autoregressive text generation."""

from typing import List, Optional, Set
import torch
import torch.nn.functional as F


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    seen_tokens: Optional[Set[int]] = None,
) -> torch.Tensor:
    """Apply repetition penalty, temperature, top-k, and nucleus (top-p) sampling.

    Args:
        logits: 1D or 2D tensor of raw unnormalized logits
        temperature: Scaling factor for sharpness
        top_k: Number of highest probability tokens to retain
        top_p: Nucleus cumulative probability threshold
        repetition_penalty: Penalty factor applied to previously sampled tokens
        seen_tokens: Set of token IDs already generated in the current sequence
    Returns:
        Sampled token ID tensor
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # 1. Repetition penalty
    if repetition_penalty != 1.0 and seen_tokens:
        for token_id in seen_tokens:
            if logits[0, token_id] < 0:
                logits[0, token_id] *= repetition_penalty
            else:
                logits[0, token_id] /= repetition_penalty

    # 2. Temperature scaling
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    # 3. Top-K filtering
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        val, _ = torch.topk(logits, k)
        logits[logits < val[:, [-1]]] = float("-inf")

    # 4. Top-P (Nucleus) filtering
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float("-inf")

    # 5. Softmax and Categorical sampling
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

