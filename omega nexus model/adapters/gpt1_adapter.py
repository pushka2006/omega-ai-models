"""OMEGA NEXUS MODEL: Adapter for OMEGA GPT-1 (Model 1).

Dynamically loads Model 1 without sys.modules collisions.
"""

from pathlib import Path
import importlib.util
import torch

from config import MODEL_1_DIR


class GPT1Adapter:
    """Interoperability wrapper around OMEGA GPT-1."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = False
        self._load()

    def _load(self):
        try:
            model_file = MODEL_1_DIR / "model.py"
            tok_file = MODEL_1_DIR / "tokenizer.py"
            if not model_file.exists() or not tok_file.exists():
                return

            # Load Tokenizer module
            spec_tok = importlib.util.spec_from_file_location("omega1_tokenizer", str(tok_file))
            mod_tok = importlib.util.module_from_spec(spec_tok)
            spec_tok.loader.exec_module(mod_tok)

            self.tokenizer = mod_tok.OmegaTokenizer()
            tok_path = MODEL_1_DIR / "tokenizer_saved"
            if tok_path.exists():
                self.tokenizer.load(str(tok_path))

            # Load Model module
            spec_model = importlib.util.spec_from_file_location("omega1_model", str(model_file))
            mod_model = importlib.util.module_from_spec(spec_model)
            spec_model.loader.exec_module(mod_model)

            # Load weights
            ckpt_best = MODEL_1_DIR / "checkpoints" / "omega_gpt1_best.pth"
            if ckpt_best.exists():
                ckpt = torch.load(str(ckpt_best), map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt)
                emb_weight = state.get("embeddings.token_embeddings.weight", state.get("tok_emb.weight"))
                vocab_size = emb_weight.shape[0] if emb_weight is not None else 1200
                d_model = emb_weight.shape[1] if emb_weight is not None else 256

                self.model = mod_model.GPTLanguageModel(
                    vocab_size=vocab_size,
                    n_embd=d_model,
                    n_layer=4,
                    n_head=4,
                    block_size=128,
                    dropout=0.0,
                )
                self.model.load_state_dict(state, strict=False)
                self.model.eval()
                self.available = True
        except Exception as e:
            self.available = False
            self.error = str(e)

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate response tokens using OMEGA GPT-1."""
        if not self.available or self.model is None or self.tokenizer is None:
            return f"[Nexus GPT-1 Fallback] Generated response for: '{prompt}'"

        token_ids = self.tokenizer.encode(prompt)
        x = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_tokens):
                logits = self.model(x)[:, -1, :] / max(temperature, 1e-5)
                probs = torch.softmax(logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_tok], dim=1)
                if next_tok.item() == 2:  # EOS token
                    break

        return self.tokenizer.decode(x[0].tolist())

    def get_token_embeddings(self, text: str) -> torch.Tensor:
        """Extract continuous token embeddings for multimodal fusion."""
        if not self.available or self.model is None or self.tokenizer is None:
            tokens = text.split()
            seq_len = max(len(tokens), 1)
            return torch.randn(1, seq_len, 256)

        token_ids = self.tokenizer.encode(text)
        x = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            if hasattr(self.model, "embeddings") and hasattr(self.model.embeddings, "token_embeddings"):
                emb = self.model.embeddings.token_embeddings(x)
            elif hasattr(self.model, "tok_emb"):
                emb = self.model.tok_emb(x)
            else:
                emb = torch.randn(1, x.size(1), 256)
        return emb
