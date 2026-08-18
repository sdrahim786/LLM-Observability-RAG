"""
Tests for the tracing layer and the analytics built on top of it.

These matter more than they look: if tracing is wrong, every number on the
dashboard is wrong, and a dashboard that lies is worse than no dashboard.
"""

import pytest

from src.observability import analytics
from src.observability.tracer import Tracer, get_tracer, set_tracer, span, trace


def test_span_is_recorded(isolated_tracer):
    with span("retrieval", query="test") as s:
        s["outputs"]["chunk_count"] = 3

    records = isolated_tracer.read_all()
    spans = [r for r in records if r["type"] == "span"]

    assert len(spans) == 1
    assert spans[0]["stage"] == "retrieval"
    assert spans[0]["inputs"]["query"] == "test"
    assert spans[0]["outputs"]["chunk_count"] == 3
    assert spans[0]["duration_ms"] >= 0


def test_spans_nest_under_a_trace(isolated_tracer):
    with trace("query", question="hello"):
        with span("retrieval"):
            with span("vectorstore_search"):
                pass

    records = isolated_tracer.read_all()
    spans = {r["stage"]: r for r in records if r["type"] == "span"}
    traces = [r for r in records if r["type"] == "trace"]

    assert len(traces) == 1
    trace_id = traces[0]["trace_id"]

    # Both spans belong to the trace, and the inner one points at the outer.
    assert spans["retrieval"]["trace_id"] == trace_id
    assert spans["vectorstore_search"]["trace_id"] == trace_id
    assert spans["vectorstore_search"]["parent_span_id"] == spans["retrieval"]["span_id"]
    assert spans["retrieval"]["parent_span_id"] is None


def test_errors_are_captured_and_reraised(isolated_tracer):
    with pytest.raises(ValueError):
        with span("embedding"):
            raise ValueError("model failed to load")

    spans = [r for r in isolated_tracer.read_all() if r["type"] == "span"]
    assert "ValueError: model failed to load" in spans[0]["error"]


def test_long_values_are_truncated(isolated_tracer):
    with span("llm", prompt="x" * 5000):
        pass

    spans = [r for r in isolated_tracer.read_all() if r["type"] == "span"]
    assert "truncated" in spans[0]["inputs"]["prompt"]
    assert len(spans[0]["inputs"]["prompt"]) < 5000


def test_tracing_failure_does_not_break_the_pipeline(tmp_path):
    """A tracer that cannot write must not take the application down."""
    tracer = Tracer(trace_dir=tmp_path / "traces")
    tracer.trace_dir = tmp_path / "nonexistent" / "deeply" / "nested"
    set_tracer(tracer)

    # Should complete silently rather than raising.
    with span("retrieval"):
        pass


def test_disabled_tracer_records_nothing(tmp_path):
    tracer = Tracer(trace_dir=tmp_path / "traces", enabled=False)
    set_tracer(tracer)

    with span("retrieval"):
        pass

    assert tracer.read_all() == []


def test_corrupt_line_is_skipped(isolated_tracer):
    with span("retrieval"):
        pass
    with open(isolated_tracer.trace_file, "a") as f:
        f.write("{not valid json\n")

    # One good record, corrupt line ignored rather than blowing up the read.
    assert len(isolated_tracer.read_all()) == 1


# --- Analytics ---

def test_latency_by_stage_aggregates(isolated_tracer):
    for _ in range(3):
        with span("retrieval"):
            pass
    with span("llm"):
        pass

    latency = analytics.latency_by_stage()

    assert latency["retrieval"]["calls"] == 3
    assert latency["llm"]["calls"] == 1
    assert latency["retrieval"]["p95_ms"] >= 0


def test_retrieval_quality_counts_empty_retrievals(isolated_tracer):
    with span("retrieval") as s:
        s["outputs"]["chunk_count"] = 4
        s["outputs"]["top_score"] = 0.82
    with span("retrieval") as s:
        s["outputs"]["chunk_count"] = 0

    quality = analytics.retrieval_quality()

    assert quality["queries"] == 2
    assert quality["empty_retrievals"] == 1
    assert quality["empty_rate"] == 0.5
    assert quality["avg_top_score"] == 0.82


def test_token_usage_sums_across_calls(isolated_tracer):
    for prompt_tokens, completion_tokens in [(100, 20), (200, 40)]:
        with span("llm") as s:
            s["outputs"]["prompt_tokens"] = prompt_tokens
            s["outputs"]["completion_tokens"] = completion_tokens

    tokens = analytics.token_usage()

    assert tokens["calls"] == 2
    assert tokens["prompt_tokens"] == 300
    assert tokens["completion_tokens"] == 60
    assert tokens["total_tokens"] == 360
    assert tokens["avg_prompt_tokens"] == 150.0


def test_error_summary_groups_by_stage(isolated_tracer):
    with pytest.raises(RuntimeError):
        with span("llm"):
            raise RuntimeError("boom")
    with span("retrieval"):
        pass

    errors = analytics.error_summary()

    assert errors["failed_spans"] == 1
    assert errors["by_stage"]["llm"] == 1
    assert errors["error_rate"] == 0.5


def test_summary_on_empty_traces_does_not_crash(isolated_tracer):
    result = analytics.summary()

    assert result["total_requests"] == 0
    assert result["retrieval"]["queries"] == 0
    assert result["tokens"]["calls"] == 0
