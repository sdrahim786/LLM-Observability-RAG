"""
Tests for the ingestion and chunking stages. Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.documents import Document

from src.ingestion.loader import load_file, load_directory
from src.chunking.splitter import chunk_documents, get_chunk_stats


# --- Ingestion ---

def test_load_txt_file(tmp_path):
    file_path = tmp_path / "note.txt"
    file_path.write_text("This is a test note about observability.")

    docs = load_file(file_path)

    assert len(docs) == 1
    assert "observability" in docs[0].page_content
    assert docs[0].metadata["source_file"] == "note.txt"
    assert docs[0].metadata["file_type"] == "txt"


def test_load_directory_skips_unsupported(tmp_path):
    (tmp_path / "note.txt").write_text("Supported file.")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG\r\n")

    docs = load_directory(tmp_path)

    assert len(docs) == 1
    assert docs[0].metadata["source_file"] == "note.txt"


def test_unsupported_extension_raises(tmp_path):
    file_path = tmp_path / "ignore.png"
    file_path.write_bytes(b"\x89PNG\r\n")

    try:
        load_file(file_path)
        assert False, "Expected ValueError for unsupported extension"
    except ValueError:
        pass


# --- Chunking ---

def test_chunking_splits_long_document():
    long_text = "This is a sentence about retrieval quality. " * 60
    doc = Document(page_content=long_text, metadata={"source_file": "long.txt"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_chunking_preserves_source_metadata():
    doc = Document(
        page_content="Some content here. " * 50,
        metadata={"source_file": "notes.pdf", "file_type": "pdf"},
    )

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)

    for chunk in chunks:
        assert chunk.metadata["source_file"] == "notes.pdf"
        assert chunk.metadata["file_type"] == "pdf"


def test_chunk_index_increments_per_source():
    doc = Document(page_content="Word " * 200, metadata={"source_file": "a.txt"})

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    indices = [c.metadata["chunk_index"] for c in chunks]

    assert indices == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size():
    doc = Document(page_content="text", metadata={})

    try:
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)
        assert False, "Expected ValueError when overlap >= chunk_size"
    except ValueError:
        pass


def test_chunk_stats_reports_totals():
    doc = Document(page_content="Sentence here. " * 40, metadata={"source_file": "a.txt"})

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    stats = get_chunk_stats(chunks)

    assert stats["total_chunks"] == len(chunks)
    assert stats["sources"]["a.txt"] == len(chunks)
    assert stats["avg_chunk_size"] > 0


def test_chunk_stats_handles_empty_input():
    stats = get_chunk_stats([])

    assert stats["total_chunks"] == 0
