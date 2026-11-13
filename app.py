
# ============================================================
# CHATBOT WEB APP
# ============================================================
#
# Run:   .\venv\Scripts\python.exe app.py
# Open:  http://127.0.0.1:8000
# ============================================================

import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

import rag_agent

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = (BASE_DIR / "outputs").resolve()

resources = None


@asynccontextmanager
async def lifespan(_app):
    global resources
    print("Loading embedding model, reranker and ChromaDB...")
    resources = rag_agent.Resources()
    print(f"Ready: {resources.collection.count()} chunks indexed.")
    yield


app = FastAPI(title="Perseverance Guide", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=4000)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health():
    return {"chunks": resources.collection.count() if resources else 0,
            "model": rag_agent.QWEN_MODEL}


@app.post("/api/chat")
def chat(request: ChatRequest):

    def events():
        try:
            for event in rag_agent.run(resources, request.session_id, request.message.strip()):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as error:  # surface errors to the UI
            yield f"data: {json.dumps({'type': 'error', 'message': str(error)})}\n\n"

    return StreamingResponse(
        iterate_in_threadpool(events()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/history/{session_id}")
def history(session_id: str):
    return rag_agent.load_full_history(session_id)


@app.delete("/api/history/{session_id}")
def clear(session_id: str):
    rag_agent.clear_history(session_id)
    return {"ok": True}


@app.get("/api/images/{document}/{filename}")
def image(document: str, filename: str):
    path = (OUTPUTS_DIR / document / "images" / filename).resolve()
    # Block path traversal outside outputs/.
    if OUTPUTS_DIR not in path.parents or not path.is_file():
        raise HTTPException(404, "Image not found")
    return FileResponse(path)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
