"""
Chunking -- Phase 2.

Takes the Documents produced by ingestion and splits them into uniform,
overlapping pieces small enough to embed and retrieve accurately.

Why this is its own step: ingestion produces wildly inconsistent sizes (a PDF
page might be 2000 characters, a short note 50). Embedding models have a fixed
context window, and retrieval quality degrades when a chunk carries too much
unrelated text alongside the relevant part. Chunking normalizes that.

Observability note: chunk statistics (count, size distribution, how many
chunks came from each source) are the first place retrieval problems show up.
A retrieval that returns garbage is often a chunking problem, not a search
problem -- which is why get_chunk_stats() exists here rather than being
bolted on later.
"""

from typing import Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Defaults tuned for a personal knowledge base. Tune empirically once
# retrieval is running: too small and answers lose context, too large and
# retrieval gets noisy.
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


def chunk_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split Documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries separators in order -- paragraph
    breaks, then newlines, then spaces, then raw characters -- so it only cuts
    mid-word as a last resort. That keeps ideas intact instead of slicing at an
    arbitrary character count.

    chunk_overlap repeats the tail of each chunk at the head of the next, so a
    sentence spanning a boundary still appears whole in at least one chunk.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than "
            f"chunk_size ({chunk_size}), otherwise chunks repeat endlessly."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with its position within its source file. Without this,
    # a retrieved chunk can be traced to a file but not to a location in it --
    # which makes debugging bad retrievals much harder.
    per_source_counter: Dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        index = per_source_counter.get(source, 0)
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        per_source_counter[source] = index + 1

    return chunks


def get_chunk_stats(chunks: List[Document]) -> Dict:
    """
    Summary stats for a batch of chunks. This is the first observability
    surface in the pipeline: if retrieval later returns irrelevant results,
    these numbers are where you look before blaming the vector search.
    """
    if not chunks:
        return {"total_chunks": 0, "sources": {}, "avg_chunk_size": 0}

    sizes = [len(c.page_content) for c in chunks]
    sources: Dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "unknown")
        sources[source] = sources.get(source, 0) + 1

    return {
        "total_chunks": len(chunks),
        "sources": sources,
        "avg_chunk_size": round(sum(sizes) / len(sizes), 1),
        "min_chunk_size": min(sizes),
        "max_chunk_size": max(sizes),
    }
