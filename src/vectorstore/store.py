"""
Vector store -- Phase 4.

ChromaDB, persisted to disk. Holds the embedded chunks so they survive between
runs and do not have to be recomputed every time the app starts.

Traced here: how many vectors go in, how long writes take, and how the
collection grows over time. A collection that keeps growing across runs when
you only ingested one folder means documents are being added twice, which is
easy to do and invisible without this number.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..embeddings.embedder import Embedder
from ..observability.tracer import span

DEFAULT_PERSIST_DIR = "chroma_db"
DEFAULT_COLLECTION = "documents"


class VectorStore:
    """
    Thin wrapper over a persistent Chroma collection.

    Chroma is used directly rather than through LangChain's vector store
    abstraction so that similarity scores are available unmodified. Those
    scores are one of the most useful things to trace, and wrappers often
    normalize or hide them.
    """

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.embedder = embedder
        self.persist_dir = str(Path(persist_dir))
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                raise ImportError(
                    "chromadb is not installed. Run:\n    pip install chromadb"
                ) from exc

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                # Cosine distance suits normalized sentence embeddings better
                # than Chroma's default L2.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(self, chunks: List[Document]) -> int:
        """Embed chunks and write them to the collection. Returns count added."""
        if not chunks:
            return 0

        with span("vectorstore_add", chunk_count=len(chunks)) as s:
            collection = self._get_collection()

            texts = [c.page_content for c in chunks]
            vectors = self.embedder.embed_documents(texts)

            # Deterministic IDs (source file + chunk index) mean re-ingesting
            # the same file overwrites rather than duplicates. Without this,
            # running ingest twice silently doubles your index.
            ids = []
            for i, chunk in enumerate(chunks):
                source = chunk.metadata.get("source_file", "unknown")
                index = chunk.metadata.get("chunk_index", i)
                ids.append(f"{source}::{index}")

            metadatas = [
                {k: v for k, v in c.metadata.items() if isinstance(v, (str, int, float, bool))}
                for c in chunks
            ]

            collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=texts,
                metadatas=metadatas,
            )

            total = collection.count()
            s["outputs"]["added"] = len(chunks)
            s["outputs"]["collection_size"] = total
            return len(chunks)

    def search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """
        Similarity search. Returns (Document, similarity) pairs, best first.

        Chroma returns cosine *distance* (lower is closer). It is converted to
        similarity (higher is better) here because every downstream consumer
        and every dashboard reads more naturally that way.
        """
        with span("vectorstore_search", query=query, k=k) as s:
            collection = self._get_collection()

            if collection.count() == 0:
                s["outputs"]["chunk_count"] = 0
                s["outputs"]["empty_collection"] = True
                return []

            query_vector = self.embedder.embed_query(query)
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=min(k, collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            docs: List[Tuple[Document, float]] = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for text, meta, distance in zip(documents, metadatas, distances):
                similarity = 1.0 - float(distance)
                docs.append((Document(page_content=text, metadata=dict(meta or {})), similarity))

            s["outputs"]["chunk_count"] = len(docs)
            s["outputs"]["top_score"] = round(docs[0][1], 4) if docs else None
            return docs

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def reset(self) -> None:
        """Drop everything. Useful when re-indexing from scratch."""
        with span("vectorstore_reset"):
            collection = self._get_collection()
            existing = collection.get(include=[])
            ids = existing.get("ids", [])
            if ids:
                collection.delete(ids=ids)

    def stats(self) -> Dict[str, Any]:
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "vectors": self.count(),
            "dimensions": getattr(self.embedder, "dimensions", None),
        }
