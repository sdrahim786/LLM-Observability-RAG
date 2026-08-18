"""
Orchestration -- Phase 7.

Ties every stage together: ingest documents into the index, and answer
questions using retrieval plus memory plus the LLM.

The whole point of this class is that one call to ask() produces one complete
trace tree: memory read, retrieval, prompt build, LLM call, memory write. Open
that trace and you can see exactly what the model was given and what it cost,
which is the question the entire project exists to answer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..chunking.splitter import chunk_documents, get_chunk_stats
from ..embeddings.embedder import Embedder, get_embedder
from ..ingestion.loader import load_directory, load_file
from ..memory.memory_store import LocalMemory
from ..observability.tracer import span, trace
from ..retrieval.retriever import Retriever
from ..vectorstore.store import VectorStore
from .llm import LLM, get_llm
from .prompts import PROMPT_VERSION, build_prompt


class Assistant:
    """
    The assembled pipeline.

    Every dependency is injected rather than constructed internally, which is
    what makes the whole thing testable offline: pass a fake embedder and a
    fake LLM and the same code path runs without a model download or a running
    Ollama server.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        llm: Optional[LLM] = None,
        memory: Optional[LocalMemory] = None,
        persist_dir: str = "chroma_db",
        memory_path: str = "memory_store.json",
        k: int = 4,
        fake: bool = False,
    ):
        self.embedder = embedder or get_embedder(fake=fake)
        self.llm = llm or get_llm(fake=fake)
        self.memory = memory or LocalMemory(path=memory_path)
        self.store = VectorStore(self.embedder, persist_dir=persist_dir)
        self.retriever = Retriever(self.store, k=k)

    # --- Ingestion side ---

    def ingest_path(self, path: str) -> Dict[str, Any]:
        """
        Load, chunk, embed and index a file or folder.

        Traced end to end so you can see how long indexing took and where it
        went, which matters as soon as a folder has more than a handful of
        documents in it.
        """
        with trace("ingest", path=str(path)) as t:
            target = Path(path)

            with span("ingestion", path=str(path)) as s:
                if target.is_dir():
                    documents = load_directory(target)
                else:
                    documents = load_file(target)
                s["outputs"]["document_count"] = len(documents)
                s["outputs"]["total_chars"] = sum(len(d.page_content) for d in documents)

            if not documents:
                t["metadata"]["result"] = "no documents found"
                return {"documents": 0, "chunks": 0, "indexed": 0}

            with span("chunking", document_count=len(documents)) as s:
                chunks = chunk_documents(documents)
                stats = get_chunk_stats(chunks)
                s["outputs"].update(stats)

            indexed = self.store.add_documents(chunks)

            t["metadata"]["documents"] = len(documents)
            t["metadata"]["chunks"] = len(chunks)

            return {
                "documents": len(documents),
                "chunks": len(chunks),
                "indexed": indexed,
                "collection_size": self.store.count(),
                "chunk_stats": stats,
            }

    # --- Query side ---

    def ask(self, question: str, remember: bool = True) -> Dict[str, Any]:
        """
        Answer a question using documents and memory.

        Order matters: memory is read before retrieval so that remembered
        context could later be used to sharpen the search query. Memory is
        written after the answer, so a fact learned in this turn is available
        on the next one but does not pollute the current retrieval.
        """
        with trace("query", question=question) as t:
            # 1. What do we already know about this user?
            memories = self.memory.search(question, limit=3)
            memory_text = self.memory.format_memories(memories)

            # 2. What do their documents say?
            results = self.retriever.retrieve(question)
            context = self.retriever.format_context(results)

            # 3. Build the prompt. Which variant fires is recorded, because
            #    "why did it refuse to answer" is usually answered right here.
            with span("prompt_build", prompt_version=PROMPT_VERSION) as s:
                prompt, prompt_name = build_prompt(question, context, memory_text)
                s["outputs"]["prompt_name"] = prompt_name
                s["outputs"]["prompt_chars"] = len(prompt)
                s["outputs"]["context_chunks"] = len(results)
                s["outputs"]["memories_used"] = len(memories)

            # 4. Generate.
            response = self.llm.generate(prompt)

            # 5. Remember the exchange for next session.
            if remember:
                self.memory.add(
                    f"User asked about: {question}",
                    kind="question",
                )

            t["metadata"].update({
                "prompt_name": prompt_name,
                "prompt_version": PROMPT_VERSION,
                "chunks_used": len(results),
                "memories_used": len(memories),
                "total_tokens": response.get("prompt_tokens", 0)
                + response.get("completion_tokens", 0),
            })

            return {
                "answer": response["text"],
                "sources": [
                    {
                        "source_file": doc.metadata.get("source_file"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "score": round(score, 4),
                        "preview": doc.page_content[:200],
                    }
                    for doc, score in results
                ],
                "memories_used": [m.get("content") for m in memories],
                "prompt_name": prompt_name,
                "prompt_version": PROMPT_VERSION,
                "tokens": {
                    "prompt": response.get("prompt_tokens", 0),
                    "completion": response.get("completion_tokens", 0),
                },
                "model": response.get("model"),
                "trace_id": t["trace_id"],
            }

    def stats(self) -> Dict[str, Any]:
        return {
            "vectors_indexed": self.store.count(),
            "memories_stored": len(self.memory.all()),
            "model": getattr(self.llm, "model", "unknown"),
            "embedder": type(self.embedder).__name__,
            "prompt_version": PROMPT_VERSION,
        }
