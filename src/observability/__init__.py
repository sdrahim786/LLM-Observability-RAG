"""Tracing and analytics for the local LLM pipeline."""

from .tracer import Tracer, get_tracer, set_tracer, span, trace
from .analytics import (
    error_summary,
    latency_by_stage,
    recent_traces,
    retrieval_quality,
    summary,
    token_usage,
)

__all__ = [
    "Tracer",
    "get_tracer",
    "set_tracer",
    "span",
    "trace",
    "summary",
    "latency_by_stage",
    "retrieval_quality",
    "token_usage",
    "error_summary",
    "recent_traces",
]
