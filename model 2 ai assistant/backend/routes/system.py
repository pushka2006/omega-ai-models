"""OMEGA MODEL 2: System Status and Health Diagnostics Endpoints."""

import os
import platform
import time
from fastapi import APIRouter
import torch

from backend.brain.llm_engine import llm_engine
from backend.brain.omega1_adapter import omega1_adapter
from backend.config import settings
from backend.database.db import get_db_connection

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/status")
def system_status():
    """Return comprehensive system health, model availability, and database telemetry."""
    # Database stats
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM memories")
    mem_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM document_chunks")
    chunk_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM conversations")
    conv_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM tool_logs")
    tool_logs_count = cursor.fetchone()[0]

    conn.close()

    # Hardware stats
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only"

    return {
        "identity": {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.DESCRIPTION,
        },
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": cuda_avail,
            "device": device_name,
        },
        "model_1_sovereign": {
            "status": omega1_adapter.metadata.get("status", "Unknown"),
            "checkpoint": settings.MODEL1_CHECKPOINT,
            "checkpoint_exists": os.path.exists(settings.MODEL1_CHECKPOINT),
            "parameters": omega1_adapter.metadata.get("parameters", 3490304),
        },
        "database_telemetry": {
            "memories_count": mem_count,
            "documents_count": doc_count,
            "chunks_count": chunk_count,
            "conversations_count": conv_count,
            "tool_executions_count": tool_logs_count,
        },
        "providers": llm_engine.list_models(),
    }

