"""OMEGA NEURAL NETWORK ENGINE: Bridge to OMEGA GPT-1 (Model 1).

Connects the general neural engine to Model 1's Transformer architecture,
token embedding matrices, and autoregressive generation.
"""

import os
import sys
from typing import Any, Dict, Optional
import torch
from config.engine_config import MODEL1_DIR


class Omega1Bridge:
    """Interoperability bridge with Model 1 (Omega GPT-1)."""

    def __init__(self):
        self.model1_dir = str(MODEL1_DIR)
        self.checkpoint_path = os.path.join(self.model1_dir, "checkpoints", "omega_gpt1_best.pth")

    def is_model1_available(self) -> bool:
        return os.path.exists(self.checkpoint_path)

    def load_model1(self, device: torch.device = torch.device("cpu")):
        """Load Model 1's Transformer model and tokenizer."""
        if self.model1_dir not in sys.path:
            sys.path.insert(0, self.model1_dir)

        from model import GPTLanguageModel
        from tokenizer import get_tokenizer

        tokenizer = get_tokenizer()
        model = GPTLanguageModel().to(device)

        if os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)

        model.eval()
        return model, tokenizer

    def extract_token_embeddings(self, text: str) -> Optional[torch.Tensor]:
        """Extract continuous token representations from Model 1's embedding layer."""
        if not self.is_model1_available():
            return None
        model, tokenizer = self.load_model1()
        tokens = tokenizer.encode_to_tensor(text)
        with torch.no_grad():
            return model.embeddings.token_embeddings(tokens)
