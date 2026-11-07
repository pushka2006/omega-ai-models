"""OMEGA MODEL 2: FastAPI Application Entrypoint.

Serves RESTful APIs, WebSocket chat streaming, and the futuristic single-page web UI.
"""

import os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routes import chat, coding, knowledge, memory, models, productivity, search, system, tools

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.DESCRIPTION,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router Modules
app.include_router(chat.router)
app.include_router(models.router)
app.include_router(memory.router)
app.include_router(knowledge.router)
app.include_router(search.router)
app.include_router(coding.router)
app.include_router(tools.router)
app.include_router(productivity.router)
app.include_router(system.router)

# Mount Static Frontend
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_index():
    """Serve the single-page futuristic OMEGA assistant UI."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "message": "Frontend UI file is initializing. Access /docs for interactive Swagger API documentation.",
    }


# WebSocket Streaming Endpoint for Real-Time Terminal & Web UI
@app.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    from backend.brain.generator_engine import generator_engine
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "")
            conv_id = data.get("conversation_id", "ws-session")
            provider = data.get("provider", "omega_offline")
            model = data.get("model", None)
            use_search = data.get("use_web_search", False)

            # Process query
            result = generator_engine.process_query(
                query=query,
                conversation_id=conv_id,
                provider=provider,
                model=model,
                use_web_search=use_search,
            )

            await websocket.send_json({
                "type": "response",
                "payload": result,
            })
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

