"""OMEGA MODEL 2: Web Search API Endpoints."""

from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from backend.brain.search_engine import search_engine

router = APIRouter(prefix="/api/search", tags=["Search"])


class SearchRequest(BaseModel):
    query: str
    max_results: int = 5


@router.post("")
def execute_search(req: SearchRequest):
    """Execute live web search and return structured snippets."""
    return search_engine.search_and_synthesize(req.query, max_results=req.max_results)

