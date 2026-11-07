"""OMEGA MODEL 2: Chat API Endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.brain.conversation_manager import conversation_manager
from backend.brain.generator_engine import generator_engine

router = APIRouter(prefix="/api/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    provider: str = "omega_offline"
    model: Optional[str] = None
    use_web_search: bool = False
    use_knowledge_base: bool = True
    temperature: float = 0.7


@router.post("")
def chat_endpoint(req: ChatRequest):
    """Process a chat prompt and return synthesized response."""
    conv_id = req.conversation_id or conversation_manager.create_conversation(title=req.query[:30] + "...")
    result = generator_engine.process_query(
        query=req.query,
        conversation_id=conv_id,
        provider=req.provider,
        model=req.model,
        use_web_search=req.use_web_search,
        use_knowledge_base=req.use_knowledge_base,
        temperature=req.temperature,
    )
    return result


@router.get("/conversations")
def list_conversations():
    """List recent conversation sessions."""
    return conversation_manager.list_conversations(limit=30)


@router.get("/conversations/{conversation_id}")
def get_conversation_history(conversation_id: str):
    """Get all messages for a specific conversation."""
    messages = conversation_manager.get_messages(conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str):
    """Delete a conversation session."""
    success = conversation_manager.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"success": True}

