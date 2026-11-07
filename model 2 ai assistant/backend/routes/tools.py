"""OMEGA MODEL 2: Tool Registry API Endpoints."""

from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from backend.brain.tool_engine import tool_engine

router = APIRouter(prefix="/api/tools", tags=["Tools"])


class ToolExecutionRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = {}


@router.get("")
def list_registered_tools():
    """List all registered tools and parameter specifications."""
    return tool_engine.list_tools()


@router.post("/execute")
def execute_tool(req: ToolExecutionRequest):
    """Execute a registered tool directly."""
    return tool_engine.execute(req.tool_name, req.arguments)

