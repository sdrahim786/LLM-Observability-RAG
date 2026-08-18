"""
Retrieval -- Phase 5.

Finds the chunks most relevant to a question. This is the highest-value trace
point in the pipeline: most reports of "the assistant made something up" are
actually retrieval failures, where nothing relevant came back and the model
answered from its own priors instead.

So this stage records more than latency. It records what came back, how
confident the match was, and explicitly whether the result was empty or weak,
because those two cases are the ones that produce bad answers downstream.
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..observability.tracer import span
from ..vectorstore.store import VectorStore

# Below this cosine similarity, a "match" is usually noise. Chunks under the
# threshold are dropped rather than passed to the LLM, since irrelevant context
# actively makes answers worse -- the model tries to use what it is given.
DEFAULT_MIN_SCORE = 0.15
DEFAULT_K = 4


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        k: int = DEFAULT_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.store = store
        self.k = k
        self.min_score = min_score

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """
        Search, then filter out weak matches.

        The trace records both the raw hit count and the count after
        filtering. When those two numbers diverge, the index has content but
        nothing that actually answers the question, which is a very different
        problem from an empty index.
        """
        k = k or self.k

        with span("retrieval", query=query, k=k, min_score=self.min_score) as s:
            raw_results = self.store.search(query, k=k)
            filtered = [(doc, score) for doc, score in raw_results if score >= self.min_score]

            s["outputs"]["raw_count"] = len(raw_results)
            s["outputs"]["chunk_count"] = len(filtered)
            s["outputs"]["filtered_out"] = len(raw_results) - len(filtered)
            s["outputs"]["top_score"] = round(filtered[0][1], 4) if filtered else None
            s["outputs"]["sources"] = [
                doc.metadata.get("source_file", "unknown") for doc, _ in filtered
            ]
            # An explicit flag beats inferring it from a zero later. This is
            # the number the dashboard alerts on.
            s["outputs"]["empty"] = len(filtered) == 0

            return filtered

    def format_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Build the context block that gets injected into the prompt.

        Sources are labelled so the model can cite them and so a human reading
        a trace can tell which chunk produced which claim.
        """
        if not results:
            return ""

        blocks = []
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source_file", "unknown")
            chunk_index = doc.metadata.get("chunk_index", "?")
            blocks.append(
                f"[{i}] source: {source} (chunk {chunk_index}, relevance {score:.2f})\n"
                f"{doc.page_content}"
            )
        return "\n\n".join(blocks)

    def stats(self) -> Dict[str, Any]:
        return {"k": self.k, "min_score": self.min_score, "indexed_vectors": self.store.count()}
