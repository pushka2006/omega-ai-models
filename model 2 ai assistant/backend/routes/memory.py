"""OMEGA MODEL 2: Memory Center Endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.brain.memory_engine import memory_engine

router = APIRouter(prefix="/api/memory", tags=["Memory"])


class AddMemoryRequest(BaseModel):
    memory_type: str = "long_term"  # 'short_term', 'long_term', 'project'
    category: str = "preference"    # 'preference', 'fact', 'architecture', 'context'
    content: str
    importance_score: float = 0.7
    expires_in_seconds: Optional[float] = None


class SearchMemoryRequest(BaseModel):
    query: str
    memory_type: Optional[str] = None
    min_score: float = 0.0
    limit: int = 15


@router.get("")
def list_memories(type: Optional[str] = None, limit: int = 50):
    """List all stored memories."""
    return memory_engine.list_all(memory_type=type, limit=limit)


@router.post("")
def add_memory(req: AddMemoryRequest):
    """Add a new memory record."""
    mem_id = memory_engine.add_memory(
        memory_type=req.memory_type,
        category=req.category,
        content=req.content,
        importance_score=req.importance_score,
        expires_in_seconds=req.expires_in_seconds,
    )
    return {"success": True, "id": mem_id}


@router.post("/search")
def search_memories(req: SearchMemoryRequest):
    """Search stored memories."""
    return memory_engine.search_memories(
        query=req.query,
        memory_type=req.memory_type,
        min_score=req.min_score,
        limit=req.limit,
    )


@router.delete("/{memory_id}")
def delete_memory(memory_id: str):
    """Delete a memory record."""
    success = memory_engine.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory item not found")
    return {"success": True}


@router.delete("")
def clear_memories(type: Optional[str] = None):
    """Clear memories."""
    memory_engine.clear_all(memory_type=type)
    return {"success": True}

