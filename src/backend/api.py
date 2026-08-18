"""
FastAPI backend.

Two kinds of endpoint here, and the split is the point of the project:

  /ingest, /query   the assistant itself
  /traces, /metrics the observability surface over it

The observability endpoints read the same trace file the pipeline writes to,
so the dashboard is never a separate source of truth. What you see is exactly
what was recorded.
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..observability import analytics
from ..observability.tracer import get_tracer
from ..orchestration.assistant import Assistant

app = FastAPI(
    title="Local LLM Observability",
    description="A local RAG assistant with full tracing over every stage.",
    version="0.1.0",
)

# One assistant for the process. FAKE_BACKENDS=1 runs it with the stand-in
# embedder and LLM, which is how the test suite and CI exercise the API
# without a model download or a running Ollama.
_assistant: Optional[Assistant] = None


def get_assistant() -> Assistant:
    global _assistant
    if _assistant is None:
        _assistant = Assistant(fake=os.environ.get("FAKE_BACKENDS") == "1")
    return _assistant


def set_assistant(assistant: Assistant) -> None:
    """Injection point for tests."""
    global _assistant
    _assistant = assistant


# --- Request/response models ---

class IngestRequest(BaseModel):
    path: str = Field(..., description="File or folder to ingest")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    remember: bool = Field(True, description="Store this exchange in long-term memory")


class MemoryRequest(BaseModel):
    content: str = Field(..., min_length=1)


# --- Assistant endpoints ---

@app.get("/health")
def health() -> Dict[str, Any]:
    assistant = get_assistant()
    llm = assistant.llm
    return {
        "status": "ok",
        "llm_available": llm.is_available() if hasattr(llm, "is_available") else None,
        **assistant.stats(),
    }


@app.post("/ingest")
def ingest(request: IngestRequest) -> Dict[str, Any]:
    try:
        return get_assistant().ingest_path(request.path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Path not found: {request.path}")
    except NotADirectoryError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/query")
def query(request: QueryRequest) -> Dict[str, Any]:
    try:
        return get_assistant().ask(request.question, remember=request.remember)
    except Exception as exc:
        # Surface the failure rather than a bare 500, and note that the trace
        # for the failed request is already on disk with the error attached.
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")


@app.get("/memories")
def list_memories() -> Dict[str, Any]:
    memories = get_assistant().memory.all()
    return {"count": len(memories), "memories": memories}


@app.post("/memories")
def add_memory(request: MemoryRequest) -> Dict[str, Any]:
    memory_id = get_assistant().memory.add(request.content, kind="manual")
    return {"id": memory_id, "content": request.content}


@app.delete("/memories")
def clear_memories() -> Dict[str, str]:
    get_assistant().memory.clear()
    return {"status": "cleared"}


# --- Observability endpoints ---

@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    """Everything the dashboard needs in one call."""
    return analytics.summary()


@app.get("/metrics/latency")
def latency() -> Dict[str, Any]:
    return analytics.latency_by_stage()


@app.get("/metrics/retrieval")
def retrieval_metrics() -> Dict[str, Any]:
    """Retrieval health, including the empty-retrieval rate."""
    return analytics.retrieval_quality()


@app.get("/metrics/tokens")
def token_metrics() -> Dict[str, Any]:
    return analytics.token_usage()


@app.get("/metrics/errors")
def error_metrics() -> Dict[str, Any]:
    return analytics.error_summary()


@app.get("/traces")
def traces(limit: int = 20) -> Dict[str, Any]:
    """Recent end-to-end requests with their spans, newest first."""
    return {"traces": analytics.recent_traces(limit=limit)}


@app.delete("/traces")
def clear_traces() -> Dict[str, str]:
    get_tracer().clear()
    return {"status": "cleared"}
