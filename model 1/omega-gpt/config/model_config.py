"""Model Configuration for Omega GPT-1.

Supports preset profiles (tiny, small, medium) and custom parameterizations.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    context_length: int = 128
    d_model: int = 256
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 1024  # 4 * d_model
    dropout: float = 0.1
    activation: str = "gelu"  # "gelu", "swiglu", "relu"
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            )
        if self.d_ff is None or self.d_ff <= 0:
            self.d_ff = 4 * self.d_model

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def estimate_parameters(self) -> int:
        """Calculate approximate number of parameters based on configuration."""
        token_emb = self.vocab_size * self.d_model
        pos_emb = self.context_length * self.d_model

        attn_params = 4 * (self.d_model**2)
        ln_params = 4 * self.d_model

        if self.activation == "swiglu":
            # SwiGLU has 3 linear projections: gate, value, out
            ffn_params = 3 * self.d_model * self.d_ff + self.d_model
        else:
            ffn_params = 2 * self.d_model * self.d_ff + self.d_ff + self.d_model

        block_params = (attn_params + ln_params + ffn_params) * self.num_layers
        final_ln = 2 * self.d_model
        lm_head = 0 if self.tie_word_embeddings else (self.vocab_size * self.d_model)

        return token_emb + pos_emb + block_params + final_ln + lm_head

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


PROFILES: Dict[str, ModelConfig] = {
    "tiny": ModelConfig(
        vocab_size=4000,
        context_length=64,
        d_model=128,
        num_layers=2,
        num_heads=2,
        d_ff=512,
        dropout=0.1,
        activation="gelu",
    ),
    "small": ModelConfig(
        vocab_size=8000,
        context_length=128,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        dropout=0.1,
        activation="gelu",
    ),
    "medium": ModelConfig(
        vocab_size=16000,
        context_length=256,
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        activation="gelu",
    ),
}


def get_model_config(profile: str = "small", **overrides) -> ModelConfig:
    """Return a ModelConfig by profile name with optional parameter overrides."""
    profile_lower = profile.lower()
    if profile_lower in PROFILES:
        base = PROFILES[profile_lower].to_dict()
    else:
        base = PROFILES["small"].to_dict()

    base.update(overrides)
    return ModelConfig(**base)
