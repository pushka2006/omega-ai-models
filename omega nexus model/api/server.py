"""OMEGA NEXUS MODEL: FastAPI REST & Orchestration Server."""

from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch

from core.orchestrator import OmegaNexus
from config import VERSION

app = FastAPI(
    title="OMEGA NEXUS MODEL API",
    description="Multimodal Master Orchestration API for the OMEGA AI Ecosystem",
    version=VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nexus = OmegaNexus()


class QueryRequest(BaseModel):
    prompt: str
    mode: Optional[str] = "auto"
    features: Optional[List[float]] = None
    user_id: Optional[str] = "nexus_user"


class FusionRequest(BaseModel):
    text: str
    features: List[float]


@app.get("/")
def get_root():
    """Health check and ecosystem status."""
    return nexus.get_system_status()


@app.post("/api/v1/nexus/query")
def query_nexus(req: QueryRequest):
    """Dispatch query across the OMEGA ecosystem."""
    feat_tensor = None
    if req.features:
        feat_tensor = torch.tensor([req.features], dtype=torch.float32)

    result = nexus.dispatch(
        prompt=req.prompt,
        features=feat_tensor,
        mode=req.mode or "auto",
        user_id=req.user_id or "nexus_user",
    )
    return result


@app.post("/api/v1/nexus/fuse")
def fuse_modalities(req: FusionRequest):
    """Execute cross-attention multimodal fusion."""
    feat_tensor = torch.tensor([req.features], dtype=torch.float32)
    return nexus.dispatch(prompt=req.text, features=feat_tensor, mode="multimodal")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="127.0.0.1", port=8600, reload=True)
