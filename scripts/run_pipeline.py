"""
Run the pipeline stages built so far (ingestion -> chunking) and print
the observability stats for each stage.

Usage:
    python scripts/run_pipeline.py data/raw
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.loader import load_directory
from src.chunking.splitter import chunk_documents, get_chunk_stats


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/raw"

    # --- Ingestion ---
    start = time.perf_counter()
    documents = load_directory(folder)
    ingest_ms = (time.perf_counter() - start) * 1000

    print(f"\n[ingestion]  {len(documents)} document(s)  {ingest_ms:.1f}ms")
    for doc in documents:
        print(f"             {doc.metadata.get('source_file')} "
              f"({len(doc.page_content)} chars)")

    if not documents:
        print("\nNo documents found. Drop files into the folder and re-run.")
        return

    # --- Chunking ---
    start = time.perf_counter()
    chunks = chunk_documents(documents)
    chunk_ms = (time.perf_counter() - start) * 1000

    stats = get_chunk_stats(chunks)
    print(f"\n[chunking]   {stats['total_chunks']} chunk(s)  {chunk_ms:.1f}ms")
    print(f"             size: avg {stats['avg_chunk_size']}, "
          f"min {stats['min_chunk_size']}, max {stats['max_chunk_size']}")
    for source, count in sorted(stats["sources"].items()):
        print(f"             {source}: {count} chunk(s)")

    print("\n--- first chunk ---")
    print(f"metadata: {chunks[0].metadata}")
    print(f"content:  {chunks[0].page_content[:200]}...")


if __name__ == "__main__":
    main()
