"""
Embeddings -- Phase 3.

Turns text into vectors so chunks can be compared by meaning rather than by
keyword overlap. Runs locally with sentence-transformers (all-MiniLM-L6-v2),
which is small, fast, and downloads once to a local cache.

Every embedding call is traced: how many texts, how long, throughput. Embedding
is usually the slowest part of ingestion, and the first place to look when
"indexing my documents takes forever".

The Embedder protocol exists so the rest of the pipeline never imports
sentence-transformers directly. That keeps the test suite runnable with a
deterministic fake and no model download.
"""

import hashlib
import math
from typing import List, Optional, Protocol, runtime_checkable

from ..observability.tracer import span

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSIONS = 384  # all-MiniLM-L6-v2 output size


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into vectors. Swap implementations freely."""

    dimensions: int

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...

    def embed_query(self, text: str) -> List[float]: ...


class LocalEmbedder:
    """
    sentence-transformers running on your machine. No API key, no network
    after the initial model download.

    The model is loaded lazily on first use rather than in __init__, so
    importing this module stays cheap and constructing the object in a test
    does not pull a few hundred megabytes off disk.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.dimensions = DEFAULT_DIMENSIONS
        self._model = None

    def _load(self):
        if self._model is None:
            with span("embedding_model_load", model=self.model_name):
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise ImportError(
                        "sentence-transformers is not installed. Run:\n"
                        "    pip install sentence-transformers"
                    ) from exc
                self._model = SentenceTransformer(self.model_name)
                self.dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with span("embedding", count=len(texts), model=self.model_name) as s:
            model = self._load()
            vectors = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            result = [v.tolist() for v in vectors]
            s["outputs"]["vectors"] = len(result)
            s["outputs"]["dimensions"] = self.dimensions
            s["outputs"]["total_chars"] = sum(len(t) for t in texts)
            return result

    def embed_query(self, text: str) -> List[float]:
        with span("embedding_query", chars=len(text)) as s:
            model = self._load()
            vector = model.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
            s["outputs"]["dimensions"] = len(vector)
            return vector.tolist()


class HashEmbedder:
    """
    Deterministic stand-in for the real model.

    Produces stable, normalized vectors from a hash of the text. Similar
    strings do NOT get similar vectors, so this is useless for real retrieval
    quality -- it exists so the pipeline, the tracing, and the vector store
    can all be tested end to end without downloading a model or needing a
    network connection.

    Never use this in production. It is wired in only where tests ask for it.
    """

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS):
        self.dimensions = dimensions

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Stretch the digest out to the required dimensionality.
        raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(self.dimensions)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        with span("embedding", count=len(texts), model="hash-fake") as s:
            vectors = [self._vector(t) for t in texts]
            s["outputs"]["vectors"] = len(vectors)
            s["outputs"]["dimensions"] = self.dimensions
            return vectors

    def embed_query(self, text: str) -> List[float]:
        with span("embedding_query", chars=len(text)) as s:
            vector = self._vector(text)
            s["outputs"]["dimensions"] = len(vector)
            return vector


def get_embedder(fake: bool = False, model_name: Optional[str] = None) -> Embedder:
    """Factory. Pass fake=True in tests to skip the model download."""
    if fake:
        return HashEmbedder()
    return LocalEmbedder(model_name or DEFAULT_MODEL)
