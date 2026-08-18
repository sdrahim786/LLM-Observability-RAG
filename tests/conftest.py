"""
Shared fixtures.

Every test gets its own tracer writing to a temp directory, so tests never
read each other's spans and never touch the real traces file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.embedder import HashEmbedder
from src.memory.memory_store import LocalMemory
from src.observability.tracer import Tracer, set_tracer
from src.orchestration.assistant import Assistant
from src.orchestration.llm import FakeLLM


@pytest.fixture(autouse=True)
def isolated_tracer(tmp_path):
    """Point the global tracer at a temp dir for the duration of each test."""
    tracer = Tracer(trace_dir=tmp_path / "traces")
    set_tracer(tracer)
    yield tracer
    set_tracer(Tracer())


@pytest.fixture
def assistant(tmp_path):
    """A fully wired assistant using stand-ins. No downloads, no Ollama."""
    return Assistant(
        embedder=HashEmbedder(),
        llm=FakeLLM(canned_response="Answer grounded in the provided context [1]."),
        memory=LocalMemory(path=str(tmp_path / "memory.json")),
        persist_dir=str(tmp_path / "chroma"),
    )


@pytest.fixture
def sample_docs(tmp_path):
    """A small corpus on disk."""
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "chunking.md").write_text(
        "# Chunking\n\n"
        "RecursiveCharacterTextSplitter splits on paragraphs first, then "
        "sentences, then words. A good starting point is chunk_size 500 with "
        "chunk_overlap 50. Overlap preserves context across boundaries.\n"
    )
    (folder / "stack.txt").write_text(
        "The embedding model is all-MiniLM-L6-v2 running locally through "
        "sentence-transformers. Vectors are stored in ChromaDB on disk. The "
        "LLM is Llama 3.1 served by Ollama.\n"
    )
    return folder
