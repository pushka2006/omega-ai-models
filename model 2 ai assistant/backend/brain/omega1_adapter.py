"""OMEGA MODEL 2: Model 1 (Omega GPT-1) Local Adapter.

Loads and performs local inference with the sovereign Omega GPT-1 language model
trained entirely from scratch in Model 1.
"""

import os
import sys
import time
from typing import Any, Dict, Generator, Optional
import torch

from backend.config import MODEL1_CHECKPOINT, MODEL1_DIR, MODEL1_TOKENIZER_DIR


class Omega1Adapter:
    """Interface adapter for the sovereign Omega GPT-1 local model."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_loaded = False
        self.load_error = None
        self.metadata = {
            "name": "Omega GPT-1",
            "provider": "local_sovereign",
            "profile": "Small (4 Layers, 4 Heads, d_model=256)",
            "parameters": 3490304,
            "vocab_size": 1200,
            "type": "Custom Decoder-Only Transformer (From Scratch)",
            "checkpoint_path": MODEL1_CHECKPOINT,
            "capabilities": ["Autoregressive text generation", "Transformer domain theory", "Research inspection"],
            "status": "Ready",
            "note": "Trained from scratch. Labeled as Experimental local model; external providers available for advanced reasoning.",
        }

    def load(self) -> bool:
        """Load Omega GPT-1 architecture and trained weights into memory."""
        if self.is_loaded:
            return True

        if not os.path.exists(MODEL1_CHECKPOINT):
            self.load_error = f"Checkpoint not found at '{MODEL1_CHECKPOINT}'."
            self.metadata["status"] = "Missing Checkpoint"
            return False

        try:
            # Ensure model 1 root is in sys.path
            m1_path = str(MODEL1_DIR)
            if m1_path not in sys.path:
                sys.path.insert(0, m1_path)

            from model import GPTLanguageModel
            from tokenizer import get_tokenizer

            # Load Tokenizer
            self.tokenizer = get_tokenizer(MODEL1_TOKENIZER_DIR)

            # Load Model Architecture & Checkpoint
            self.model = GPTLanguageModel(
                vocab_size=self.tokenizer.get_vocab_size(),
                block_size=128,
                n_embd=256,
                n_head=4,
                n_layer=4,
                d_ff=1024,
                dropout=0.0,
                activation="gelu",
                tie_weights=True,
            ).to(self.device)

            ckpt_data = torch.load(MODEL1_CHECKPOINT, map_location=self.device, weights_only=False)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()

            self.is_loaded = True
            self.metadata["status"] = "Operational (Loaded)"
            self.metadata["step"] = ckpt_data.get("step", 250)
            self.metadata["val_loss"] = ckpt_data.get("val_loss", 3.2356)
            return True

        except Exception as e:
            self.load_error = str(e)
            self.metadata["status"] = f"Error: {str(e)[:50]}"
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """Generate response synchronously."""
        if not self.is_loaded:
            if not self.load():
                return {
                    "text": f"[Omega GPT-1 Unavailable: {self.load_error}]",
                    "model": "omega-gpt1",
                    "tokens": 0,
                    "tok_per_sec": 0,
                }

        t0 = time.time()
        idx = self.tokenizer.encode_to_tensor(prompt, device=self.device)
        if idx.shape[1] == 0:
            idx = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            output_tensor = self.model.generate(
                idx,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.1,
            )

        elapsed = time.time() - t0
        output_ids = output_tensor[0].tolist()
        gen_text = self.tokenizer.decode(output_ids)
        tokens_count = len(output_ids) - idx.shape[1]
        speed = tokens_count / elapsed if elapsed > 0 else 0

        return {
            "text": gen_text,
            "model": "omega-gpt1",
            "tokens": tokens_count,
            "tok_per_sec": round(speed, 1),
            "device": self.device,
        }

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Generator[str, None, None]:
        """Stream generated subwords piece by piece."""
        if not self.is_loaded:
            if not self.load():
                yield f"[Omega GPT-1 Unavailable: {self.load_error}]"
                return

        idx = self.tokenizer.encode_to_tensor(prompt, device=self.device)
        if idx.shape[1] == 0:
            idx = torch.tensor([[self.tokenizer.bos_id]], dtype=torch.long, device=self.device)

        curr_idx = idx
        block_size = 128

        for _ in range(max_tokens):
            idx_cond = curr_idx if curr_idx.size(1) <= block_size else curr_idx[:, -block_size:]
            with torch.no_grad():
                logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Temperature & Top-K / Top-P
            if temperature > 0.0:
                logits = logits / max(temperature, 1e-6)
                if top_k > 0:
                    val = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, val, dim=-1)[0][:, -1, None]
                    logits[logits < threshold] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = next_token.item()
            curr_idx = torch.cat((curr_idx, next_token), dim=1)

            piece = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield piece

            if hasattr(self.tokenizer, "eos_id") and token_id == self.tokenizer.eos_id:
                break


# Singleton instance
omega1_adapter = Omega1Adapter()

