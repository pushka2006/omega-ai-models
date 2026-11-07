"""OMEGA MODEL 2: Model Manager Endpoints."""

from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel

from backend.brain.llm_engine import llm_engine
from backend.brain.omega1_adapter import omega1_adapter

router = APIRouter(prefix="/api/models", tags=["Models"])


class ModelTestRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    prompt: str = "Explain the attention mechanism in Transformers."


@router.get("")
def list_available_models():
    """Return all configured and available LLM models and local providers."""
    return {
        "models": llm_engine.list_models(),
        "omega_gpt1_metadata": omega1_adapter.metadata,
    }


@router.post("/test")
def test_model(req: ModelTestRequest):
    """Run an isolated test inference on a specific model provider."""
    output = llm_engine.generate(
        prompt=req.prompt,
        provider=req.provider,
        model=req.model,
        max_tokens=60,
    )
    return output

