"""Production text generation engine for Omega GPT-1 with streaming support."""

import os
import time
from typing import Any, Dict, Generator, Optional
import torch
import torch.nn.functional as F

from config.model_config import ModelConfig
from model.gpt import OmegaGPT
from model.model_utils import count_parameters, format_parameter_count
from tokenizer.tokenizer_utils import OmegaTokenizer


class TextGenerator:
    """Production inference and generation class for Omega GPT-1."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        tokenizer_dir: str = "tokenizer/saved",
        device: str = "auto",
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load Tokenizer
        self.tokenizer = OmegaTokenizer.load(tokenizer_dir)

        # Checkpoint restoration
        self.checkpoint_info = {}
        self.checkpoint_path = checkpoint_path

        if self.checkpoint_path is None:
            candidates = [
                "outputs/checkpoints/omega_gpt1_best.pth",
                "outputs/checkpoints/omega_gpt1_latest.pth",
                "checkpoints/gpt_model.pth",
                "gpt_model.pth",
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    self.checkpoint_path = cand
                    break

        if self.checkpoint_path is not None and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                model_cfg_dict = ckpt.get("model_config", {})
                self.model_config = ModelConfig.from_dict(model_cfg_dict)
                self.model = OmegaGPT(self.model_config).to(self.device)
                self.model.load_state_dict(ckpt["model_state_dict"])
                self.checkpoint_info = {
                    "path": self.checkpoint_path,
                    "step": ckpt.get("step", 0),
                    "val_loss": ckpt.get("val_loss", None),
                }
            else:
                # Direct state dict
                self.model_config = ModelConfig(vocab_size=self.tokenizer.get_vocab_size())
                self.model = OmegaGPT(self.model_config).to(self.device)
                self.model.load_state_dict(ckpt)
                self.checkpoint_info = {"path": self.checkpoint_path, "step": "N/A", "val_loss": None}
        else:
            self.model_config = ModelConfig(vocab_size=self.tokenizer.get_vocab_size())
            self.model = OmegaGPT(self.model_config).to(self.device)
            self.checkpoint_info = {"path": "None (Random Initialization)", "step": 0, "val_loss": None}

        self.model.eval()
        self.total_params, self.trainable_params = count_parameters(self.model)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> Generator[str, None, Dict[str, Any]]:
        """Streaming text generation yielding token chunks in real time."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]

        idx = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        generated_count = 0
        t_start = time.time()
        prev_decoded = self.tokenizer.decode(token_ids)

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.model_config.context_length else idx[:, -self.model_config.context_length:]

            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen_tokens = set(idx[b].tolist())
                    for t in seen_tokens:
                        if logits[b, t] < 0:
                            logits[b, t] *= repetition_penalty
                        else:
                            logits[b, t] /= repetition_penalty

            # Temperature
            if temperature > 0.0:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            idx = torch.cat((idx, idx_next), dim=1)
            generated_count += 1

            # Decode progressive token text
            current_decoded = self.tokenizer.decode(idx[0].tolist())
            new_text_chunk = current_decoded[len(prev_decoded):]
            prev_decoded = current_decoded

            yield new_text_chunk

            if idx_next.item() == self.tokenizer.eos_id:
                break

        duration = time.time() - t_start
        speed = generated_count / max(duration, 1e-5)

        return {
            "prompt": prompt,
            "generated_text": self.tokenizer.decode(idx[0].tolist()),
            "tokens_generated": generated_count,
            "duration_seconds": duration,
            "tokens_per_second": speed,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate complete text completion from a prompt."""
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not token_ids:
            token_ids = [self.tokenizer.bos_id]

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        t_start = time.time()
        output_ids = self.model.generate(
            idx=input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_id=self.tokenizer.eos_id,
        )
        duration = time.time() - t_start

        generated_token_count = output_ids.size(1) - input_tensor.size(1)
        speed = generated_token_count / max(duration, 1e-5)
        full_text = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        result = {
            "prompt": prompt,
            "generated_text": full_text,
            "tokens_generated": generated_token_count,
            "duration_seconds": duration,
            "tokens_per_second": speed,
        }

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Tokens: {generated_token_count} in {duration:.2f}s ({speed:.1f} tok/s)\n")
                f.write("=" * 50 + "\n")
                f.write(full_text)

        return result
