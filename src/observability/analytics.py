"""
Turns raw spans into the numbers you actually want to look at.

The tracer records what happened. This module answers questions about it:
where is time going, how often does retrieval come back empty, what is the
token cost per query, which stage fails most.

Kept separate from tracer.py on purpose. Writing traces has to be fast and
never fail; reading and aggregating them can be slower and is allowed to be
opinionated about what matters.
"""

from statistics import mean, median
from typing import Any, Dict, List, Optional

from .tracer import get_tracer


def _percentile(values: List[float], pct: float) -> float:
    """Nearest-rank percentile. Avoids a numpy dependency for one function."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(pct / 100 * len(ordered) + 0.5)) - 1))
    return ordered[index]


def load_spans(stage: Optional[str] = None) -> List[Dict[str, Any]]:
    spans = [r for r in get_tracer().read_all() if r.get("type") == "span"]
    if stage:
        spans = [s for s in spans if s.get("stage") == stage]
    return spans


def load_traces() -> List[Dict[str, Any]]:
    return [r for r in get_tracer().read_all() if r.get("type") == "trace"]


def latency_by_stage() -> Dict[str, Dict[str, float]]:
    """
    Where time goes, broken down by stage.

    p95 matters more than the average here. An LLM call that is usually fast
    but occasionally stalls shows up in p95 and disappears in the mean, and
    the stall is what users actually notice.
    """
    by_stage: Dict[str, List[float]] = {}
    for s in load_spans():
        by_stage.setdefault(s.get("stage", "unknown"), []).append(s.get("duration_ms", 0))

    return {
        stage: {
            "calls": len(durations),
            "avg_ms": round(mean(durations), 2),
            "median_ms": round(median(durations), 2),
            "p95_ms": round(_percentile(durations, 95), 2),
            "max_ms": round(max(durations), 2),
            "total_ms": round(sum(durations), 2),
        }
        for stage, durations in sorted(by_stage.items())
    }


def retrieval_quality() -> Dict[str, Any]:
    """
    Retrieval health. This is the highest-value view in the whole dashboard.

    Empty retrievals are the single most common cause of an assistant making
    something up: nothing relevant came back, so the model answered from its
    own priors instead. That number being non-zero is a bug, not a statistic.
    """
    spans = load_spans("retrieval")
    if not spans:
        return {"queries": 0, "empty_retrievals": 0, "empty_rate": 0.0}

    chunk_counts = []
    top_scores = []
    empty = 0

    for s in spans:
        outputs = s.get("outputs", {})
        count = outputs.get("chunk_count", 0)
        chunk_counts.append(count)
        if count == 0:
            empty += 1
        top = outputs.get("top_score")
        if isinstance(top, (int, float)):
            top_scores.append(top)

    result = {
        "queries": len(spans),
        "empty_retrievals": empty,
        "empty_rate": round(empty / len(spans), 3),
        "avg_chunks_returned": round(mean(chunk_counts), 2) if chunk_counts else 0,
    }
    if top_scores:
        result["avg_top_score"] = round(mean(top_scores), 4)
        result["weakest_top_score"] = round(min(top_scores), 4)
    return result


def token_usage() -> Dict[str, Any]:
    """
    Token counts across LLM calls.

    Nothing is billed here since the model runs locally, but tokens are still
    the clearest proxy for how much work each request costs, and prompt bloat
    from over-large retrieved context shows up here first.
    """
    spans = load_spans("llm")
    if not spans:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt = sum(s.get("outputs", {}).get("prompt_tokens", 0) for s in spans)
    completion = sum(s.get("outputs", {}).get("completion_tokens", 0) for s in spans)

    return {
        "calls": len(spans),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "avg_prompt_tokens": round(prompt / len(spans), 1),
        "avg_completion_tokens": round(completion / len(spans), 1),
    }


def error_summary() -> Dict[str, Any]:
    spans = load_spans()
    failed = [s for s in spans if s.get("error")]
    by_stage: Dict[str, int] = {}
    for s in failed:
        by_stage[s.get("stage", "unknown")] = by_stage.get(s.get("stage", "unknown"), 0) + 1

    return {
        "total_spans": len(spans),
        "failed_spans": len(failed),
        "error_rate": round(len(failed) / len(spans), 4) if spans else 0.0,
        "by_stage": by_stage,
        "recent": [
            {"stage": s.get("stage"), "error": s.get("error"), "timestamp": s.get("timestamp")}
            for s in failed[-5:]
        ],
    }


def recent_traces(limit: int = 20) -> List[Dict[str, Any]]:
    """Most recent end-to-end requests, newest first, with their spans attached."""
    traces = load_traces()[-limit:]
    spans = load_spans()

    by_trace: Dict[str, List[Dict[str, Any]]] = {}
    for s in spans:
        by_trace.setdefault(s.get("trace_id"), []).append(s)

    enriched = []
    for t in reversed(traces):
        trace_spans = by_trace.get(t.get("trace_id"), [])
        enriched.append({
            **t,
            "spans": [
                {
                    "stage": s.get("stage"),
                    "duration_ms": s.get("duration_ms"),
                    "outputs": s.get("outputs", {}),
                    "error": s.get("error"),
                }
                for s in trace_spans
            ],
        })
    return enriched


def summary() -> Dict[str, Any]:
    """Everything the dashboard needs in one call."""
    traces = load_traces()
    durations = [t.get("duration_ms", 0) for t in traces]

    return {
        "total_requests": len(traces),
        "avg_request_ms": round(mean(durations), 2) if durations else 0,
        "p95_request_ms": round(_percentile(durations, 95), 2) if durations else 0,
        "latency_by_stage": latency_by_stage(),
        "retrieval": retrieval_quality(),
        "tokens": token_usage(),
        "errors": error_summary(),
    }
