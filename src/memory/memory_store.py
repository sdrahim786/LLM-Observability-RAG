"""
Memory -- Phase 6.

Long-term memory across sessions. This is deliberately NOT the vector store.
The vector store holds document content; memory holds facts about the user
("prefers concise answers", "works on a RAG project"). Conflating the two is
the most common bug in memory-augmented RAG, so they are separate modules with
separate traces and never share storage.

Two implementations:

  LocalMemory  -- JSON on disk, keyword-scored recall. No LLM required, so it
                  runs offline and in tests. This is the default.
  Mem0Memory   -- wraps mem0 when installed, which uses an LLM to extract and
                  consolidate facts automatically. Better quality, heavier
                  dependency.

Both satisfy the same interface, so the orchestration layer does not care
which one is active.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..observability.tracer import span

DEFAULT_MEMORY_PATH = "memory_store.json"
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "my", "i", "me",
    "you", "it", "this", "that", "what", "how", "do", "does", "did", "can",
}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


class LocalMemory:
    """
    Keyword-scored memory in a JSON file.

    Recall is intentionally simple: overlap between the query's keywords and
    each stored memory's keywords. It is not semantic, so it will miss
    paraphrases. That tradeoff buys zero dependencies and full offline
    operation, which matters more while the rest of the pipeline is being
    built. Swap in Mem0Memory when you want semantic recall.
    """

    def __init__(self, path: str = DEFAULT_MEMORY_PATH, user_id: str = "default"):
        self.path = Path(path)
        self.user_id = user_id
        self._memories: Optional[List[Dict[str, Any]]] = None

    def _load(self) -> List[Dict[str, Any]]:
        if self._memories is None:
            if self.path.exists():
                try:
                    with open(self.path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._memories = data.get(self.user_id, [])
                except (json.JSONDecodeError, OSError):
                    self._memories = []
            else:
                self._memories = []
        return self._memories

    def _persist(self) -> None:
        data: Dict[str, Any] = {}
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                data = {}
        data[self.user_id] = self._memories or []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add(self, content: str, **metadata) -> str:
        """Store a fact. Returns its id."""
        with span("memory_write", content=content) as s:
            memories = self._load()
            memory_id = uuid.uuid4().hex[:8]
            memories.append({
                "id": memory_id,
                "content": content,
                "created_at": time.time(),
                "metadata": metadata,
            })
            self._persist()
            s["outputs"]["memory_id"] = memory_id
            s["outputs"]["total_memories"] = len(memories)
            return memory_id

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Recall memories relevant to a query, best match first."""
        with span("memory_read", query=query, limit=limit) as s:
            memories = self._load()
            query_words = _keywords(query)

            scored = []
            for memory in memories:
                memory_words = _keywords(memory["content"])
                if not memory_words:
                    continue
                overlap = len(query_words & memory_words)
                if overlap:
                    scored.append((overlap / len(memory_words | query_words), memory))

            scored.sort(key=lambda pair: pair[0], reverse=True)
            hits = [
                {**memory, "score": round(score, 4)}
                for score, memory in scored[:limit]
            ]

            s["outputs"]["hit_count"] = len(hits)
            s["outputs"]["total_memories"] = len(memories)
            s["outputs"]["top_score"] = hits[0]["score"] if hits else None
            return hits

    def all(self) -> List[Dict[str, Any]]:
        return list(self._load())

    def clear(self) -> None:
        self._memories = []
        self._persist()

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        return "\n".join(f"- {m['content']}" for m in memories)


class Mem0Memory:
    """
    mem0-backed memory. Uses an LLM to extract and consolidate facts, which
    gives semantic recall and automatic deduplication.

    Requires mem0ai installed and a configured LLM backend. Falls back loudly
    rather than silently so a misconfiguration is not mistaken for "the
    assistant just does not remember things".
    """

    def __init__(self, user_id: str = "default", config: Optional[Dict[str, Any]] = None):
        self.user_id = user_id
        try:
            from mem0 import Memory
        except ImportError as exc:
            raise ImportError(
                "mem0ai is not installed. Run:\n    pip install mem0ai\n"
                "Or use LocalMemory, which needs no LLM."
            ) from exc
        self._memory = Memory.from_config(config) if config else Memory()

    def add(self, content: str, **metadata) -> str:
        with span("memory_write", content=content, backend="mem0") as s:
            result = self._memory.add(content, user_id=self.user_id, metadata=metadata or None)
            s["outputs"]["result"] = str(result)[:200]
            return str(result)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        with span("memory_read", query=query, limit=limit, backend="mem0") as s:
            raw = self._memory.search(query, user_id=self.user_id, limit=limit)
            results = raw.get("results", raw) if isinstance(raw, dict) else raw
            hits = [
                {
                    "id": r.get("id", ""),
                    "content": r.get("memory", r.get("text", "")),
                    "score": r.get("score"),
                }
                for r in (results or [])
            ]
            s["outputs"]["hit_count"] = len(hits)
            s["outputs"]["top_score"] = hits[0]["score"] if hits else None
            return hits

    def all(self) -> List[Dict[str, Any]]:
        raw = self._memory.get_all(user_id=self.user_id)
        return raw.get("results", raw) if isinstance(raw, dict) else raw

    def clear(self) -> None:
        self._memory.delete_all(user_id=self.user_id)

    def format_memories(self, memories: List[Dict[str, Any]]) -> str:
        if not memories:
            return ""
        return "\n".join(f"- {m.get('content', '')}" for m in memories)


def get_memory(backend: str = "local", **kwargs):
    """Factory. backend='local' needs nothing; backend='mem0' needs mem0ai."""
    if backend == "mem0":
        return Mem0Memory(**kwargs)
    return LocalMemory(**kwargs)
