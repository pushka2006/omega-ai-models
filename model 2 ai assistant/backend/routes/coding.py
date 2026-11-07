"""OMEGA MODEL 2: Coding Assistant API Endpoints."""

from typing import Any, Dict
from fastapi import APIRouter
from pydantic import BaseModel

from backend.brain.coding_engine import coding_engine

router = APIRouter(prefix="/api/coding", tags=["Coding"])


class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"


class CodeExplainRequest(BaseModel):
    code: str
    language: str = "python"


@router.post("/execute")
def execute_code(req: CodeExecutionRequest):
    """Execute Python code in an isolated sandbox."""
    if req.language.lower() != "python":
        return {"success": False, "error": f"Language '{req.language}' execution is not sandboxed locally."}
    return coding_engine.execute_python_sandbox(req.code)


@router.post("/explain")
def explain_code(req: CodeExplainRequest):
    """Analyze code structure and explain syntax."""
    explanation = coding_engine.generate_code_explanation(req.code, req.language)
    return {"explanation": explanation}


@router.post("/tests")
def generate_tests(req: CodeExplainRequest):
    """Synthesize automated unit tests for code."""
    tests = coding_engine.generate_tests_for_code(req.code, req.language)
    return {"tests": tests}

