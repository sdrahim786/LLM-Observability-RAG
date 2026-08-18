"""
Tests for the individual pipeline stages: embeddings, vector store,
retrieval, memory, and prompt construction.
"""

import pytest
from langchain_core.documents import Document

from src.embeddings.embedder import HashEmbedder
from src.memory.memory_store import LocalMemory
from src.orchestration.prompts import build_prompt
from src.retrieval.retriever import Retriever
from src.vectorstore.store import VectorStore


# --- Embeddings ---

def test_embedder_is_deterministic():
    embedder = HashEmbedder()
    assert embedder.embed_query("hello") == embedder.embed_query("hello")


def test_embedder_returns_correct_dimensions():
    embedder = HashEmbedder(dimensions=384)
    vectors = embedder.embed_documents(["one", "two"])

    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)


def test_embedding_is_traced(isolated_tracer):
    HashEmbedder().embed_documents(["a", "b", "c"])

    spans = [r for r in isolated_tracer.read_all() if r.get("stage") == "embedding"]
    assert spans[0]["outputs"]["vectors"] == 3


# --- Vector store ---

def test_add_and_search(tmp_path):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    chunks = [
        Document(page_content="alpha content", metadata={"source_file": "a.txt", "chunk_index": 0}),
        Document(page_content="beta content", metadata={"source_file": "b.txt", "chunk_index": 0}),
    ]

    added = store.add_documents(chunks)
    assert added == 2
    assert store.count() == 2

    results = store.search("alpha content", k=2)
    assert len(results) == 2
    # Exact match should score highest with a deterministic embedder.
    assert results[0][0].page_content == "alpha content"


def test_reingesting_same_file_does_not_duplicate(tmp_path):
    """Deterministic IDs mean a second ingest overwrites instead of doubling."""
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    chunks = [
        Document(page_content="content", metadata={"source_file": "a.txt", "chunk_index": 0}),
    ]

    store.add_documents(chunks)
    store.add_documents(chunks)

    assert store.count() == 1


def test_search_on_empty_store_returns_nothing(tmp_path):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    assert store.search("anything") == []


def test_add_empty_list_is_a_noop(tmp_path):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    assert store.add_documents([]) == 0


# --- Retrieval ---

def test_retriever_filters_weak_matches(tmp_path, isolated_tracer):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    store.add_documents([
        Document(page_content="exact query text", metadata={"source_file": "a.txt", "chunk_index": 0}),
    ])

    # A threshold above any achievable score filters everything out.
    retriever = Retriever(store, min_score=0.99)
    results = retriever.retrieve("something completely different")

    assert results == []

    spans = [r for r in isolated_tracer.read_all() if r.get("stage") == "retrieval"]
    assert spans[0]["outputs"]["empty"] is True
    assert spans[0]["outputs"]["raw_count"] >= spans[0]["outputs"]["chunk_count"]


def test_retrieval_records_sources(tmp_path, isolated_tracer):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    store.add_documents([
        Document(page_content="findme", metadata={"source_file": "notes.md", "chunk_index": 2}),
    ])

    retriever = Retriever(store, min_score=-1.0)
    retriever.retrieve("findme")

    spans = [r for r in isolated_tracer.read_all() if r.get("stage") == "retrieval"]
    assert "notes.md" in spans[0]["outputs"]["sources"]


def test_format_context_labels_sources(tmp_path):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    retriever = Retriever(store)
    results = [
        (Document(page_content="text here", metadata={"source_file": "a.txt", "chunk_index": 1}), 0.9),
    ]

    context = retriever.format_context(results)

    assert "[1]" in context
    assert "a.txt" in context
    assert "text here" in context


def test_format_context_empty_is_empty_string(tmp_path):
    store = VectorStore(HashEmbedder(), persist_dir=str(tmp_path / "chroma"))
    assert Retriever(store).format_context([]) == ""


# --- Memory ---

def test_memory_roundtrip(tmp_path):
    memory = LocalMemory(path=str(tmp_path / "mem.json"))
    memory.add("User prefers concise answers")

    hits = memory.search("concise")

    assert len(hits) == 1
    assert "concise" in hits[0]["content"]


def test_memory_persists_across_instances(tmp_path):
    path = str(tmp_path / "mem.json")
    LocalMemory(path=path).add("User works on a RAG project")

    reopened = LocalMemory(path=path)
    assert len(reopened.all()) == 1


def test_memory_search_is_traced(tmp_path, isolated_tracer):
    memory = LocalMemory(path=str(tmp_path / "mem.json"))
    memory.add("something memorable")
    memory.search("memorable")

    stages = [r.get("stage") for r in isolated_tracer.read_all()]
    assert "memory_write" in stages
    assert "memory_read" in stages


def test_memory_ignores_stopwords(tmp_path):
    memory = LocalMemory(path=str(tmp_path / "mem.json"))
    memory.add("User likes Python")

    # "the" and "is" alone should not match anything.
    assert memory.search("the is a") == []


def test_memory_clear(tmp_path):
    memory = LocalMemory(path=str(tmp_path / "mem.json"))
    memory.add("fact")
    memory.clear()

    assert memory.all() == []


# --- Prompts ---

def test_rag_prompt_used_when_context_exists():
    prompt, name = build_prompt("What is X?", "some context", "")

    assert name == "rag"
    assert "some context" in prompt
    assert "What is X?" in prompt


def test_no_context_prompt_used_when_retrieval_empty():
    prompt, name = build_prompt("What is X?", "", "")

    assert name == "no_context"
    assert "No relevant documents" in prompt
    # The instruction not to fall back on general knowledge is the whole point
    # of having a separate prompt for this case.
    assert "Do not answer from general knowledge" in prompt


def test_memories_are_injected_when_present():
    prompt, _ = build_prompt("Q?", "context", "- User prefers short answers")

    assert "User prefers short answers" in prompt
    assert "What you remember about this user" in prompt


def test_memory_block_absent_when_no_memories():
    prompt, _ = build_prompt("Q?", "context", "")

    assert "What you remember about this user" not in prompt
