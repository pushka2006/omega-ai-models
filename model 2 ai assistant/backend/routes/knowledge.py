"""OMEGA MODEL 2: Knowledge Hub Endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.brain.retrieval_engine import knowledge_engine

router = APIRouter(prefix="/api/knowledge", tags=["Knowledge"])


class IngestTextRequest(BaseModel):
    filename: str
    content: str
    file_type: str = "txt"


class SearchKnowledgeRequest(BaseModel):
    query: str
    top_k: int = 5


@router.get("/documents")
def list_documents():
    """List all ingested documents and chunk stats."""
    return knowledge_engine.list_documents()


@router.post("/ingest-text")
def ingest_text_endpoint(req: IngestTextRequest):
    """Ingest a text document into the knowledge base."""
    if not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
    doc_id = knowledge_engine.ingest_text(req.filename, req.content, req.file_type)
    return {"success": True, "document_id": doc_id, "filename": req.filename}


@router.post("/search")
def search_knowledge_endpoint(req: SearchKnowledgeRequest):
    """Retrieve top-scoring semantic chunks for a query."""
    return knowledge_engine.search(req.query, top_k=req.top_k)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    """Delete document and its chunk vectors."""
    success = knowledge_engine.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"success": True}

