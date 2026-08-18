"""
End-to-end tests: the full assistant, and the API on top of it.

The assertions here are mostly about the trace tree. A pipeline that returns
an answer but records nothing useful about how it got there is exactly what
this project exists to prevent.
"""

from fastapi.testclient import TestClient

from src.backend.api import app, set_assistant
from src.observability import analytics


def test_ingest_then_ask(assistant, sample_docs, isolated_tracer):
    result = assistant.ingest_path(str(sample_docs))

    assert result["documents"] == 2
    assert result["chunks"] >= 2
    assert result["collection_size"] == result["chunks"]

    answer = assistant.ask("What chunk size should I use?")

    assert answer["answer"]
    assert answer["trace_id"]
    assert answer["prompt_version"] == "v1"


def test_one_question_produces_one_complete_trace(assistant, sample_docs, isolated_tracer):
    assistant.ingest_path(str(sample_docs))
    isolated_tracer.clear()  # drop ingestion spans, keep the query clean

    assistant.ask("What embedding model is used?")

    records = isolated_tracer.read_all()
    traces = [r for r in records if r["type"] == "trace"]
    spans = [r for r in records if r["type"] == "span"]
    stages = {s["stage"] for s in spans}

    assert len(traces) == 1
    # Every stage of the query path reported in.
    assert {"memory_read", "retrieval", "prompt_build", "llm", "memory_write"} <= stages
    # And every span belongs to that one trace.
    assert all(s["trace_id"] == traces[0]["trace_id"] for s in spans)


def test_unanswerable_question_is_visible_in_traces(assistant, sample_docs, isolated_tracer):
    """
    The case that matters most: nothing relevant was retrieved. The pipeline
    should record that explicitly and switch to the no-context prompt rather
    than quietly letting the model improvise.
    """
    assistant.ingest_path(str(sample_docs))
    assistant.retriever.min_score = 0.99  # force everything to be filtered out

    answer = assistant.ask("What is the capital of France?")

    assert answer["sources"] == []
    assert answer["prompt_name"] == "no_context"

    quality = analytics.retrieval_quality()
    assert quality["empty_retrievals"] >= 1


def test_memory_carries_across_questions(assistant, sample_docs):
    assistant.ingest_path(str(sample_docs))
    assistant.memory.add("User prefers concise answers about chunking")

    answer = assistant.ask("Tell me about chunking")

    assert any("concise" in m for m in answer["memories_used"])


def test_ingest_is_traced_as_its_own_request(assistant, sample_docs, isolated_tracer):
    assistant.ingest_path(str(sample_docs))

    records = isolated_tracer.read_all()
    traces = [r for r in records if r["type"] == "trace"]
    stages = {r["stage"] for r in records if r["type"] == "span"}

    assert traces[0]["name"] == "ingest"
    assert {"ingestion", "chunking", "embedding", "vectorstore_add"} <= stages


def test_stats_reflect_state(assistant, sample_docs):
    assistant.ingest_path(str(sample_docs))
    stats = assistant.stats()

    assert stats["vectors_indexed"] > 0
    assert stats["embedder"] == "HashEmbedder"


# --- API ---

def test_api_endpoints(assistant, sample_docs, isolated_tracer):
    set_assistant(assistant)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ingest = client.post("/ingest", json={"path": str(sample_docs)})
    assert ingest.status_code == 200
    assert ingest.json()["chunks"] > 0

    query = client.post("/query", json={"question": "What chunk size?"})
    assert query.status_code == 200
    assert "answer" in query.json()

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.json()["total_requests"] >= 2

    traces = client.get("/traces", params={"limit": 5})
    assert traces.status_code == 200
    assert len(traces.json()["traces"]) >= 1


def test_api_returns_404_for_missing_path(assistant, isolated_tracer):
    set_assistant(assistant)
    client = TestClient(app)

    response = client.post("/ingest", json={"path": "/nonexistent/folder"})
    assert response.status_code in (400, 404)


def test_api_rejects_empty_question(assistant, isolated_tracer):
    set_assistant(assistant)
    client = TestClient(app)

    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
