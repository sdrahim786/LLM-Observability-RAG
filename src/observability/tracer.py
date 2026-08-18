"""
The tracing layer. This is the core of the project.

Every stage of the pipeline reports what it did through this module: what went
in, what came out, how long it took, and whether it failed. Traces are written
to a local JSONL file. No server, no cloud account, no network calls.

Design notes
------------
A *trace* is one end-to-end request (a user asks a question).
A *span* is one stage inside that request (retrieval, embedding, LLM call).
Spans nest, so a trace forms a tree and you can see where time actually went.

JSONL was chosen over SQLite deliberately: traces are append-only, and a plain
text file can be tailed, grepped, and inspected without any tooling. If volume
ever outgrows that, the reader interface here is the only thing that needs to
change.

Nothing in this module raises into the caller. Observability that can crash the
thing it observes is worse than no observability, so every failure path here
degrades to silence.
"""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_TRACE_DIR = Path(os.environ.get("TRACE_DIR", "traces"))
DEFAULT_TRACE_FILE = "traces.jsonl"

# Trace/span context for the current thread. Using thread-local rather than a
# global means concurrent API requests do not interleave their spans.
_context = threading.local()

# Values longer than this get truncated before being written. Traces are for
# diagnosis, not archival -- storing whole documents would make the file
# unusable within a day of real use.
MAX_VALUE_CHARS = 2000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value: Any) -> Any:
    """Shrink oversized values so the trace file stays readable."""
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + f"... [truncated {len(value) - MAX_VALUE_CHARS} chars]"
    if isinstance(value, list) and len(value) > 20:
        return [_truncate(v) for v in value[:20]] + [f"... [{len(value) - 20} more]"]
    if isinstance(value, dict):
        return {k: _truncate(v) for k, v in value.items()}
    return value


class Tracer:
    """
    Writes spans to a local JSONL file.

    One Tracer instance is shared process-wide (see get_tracer). Writes are
    guarded by a lock so spans from concurrent requests do not interleave
    mid-line.
    """

    def __init__(self, trace_dir: Path = DEFAULT_TRACE_DIR, enabled: bool = True):
        self.trace_dir = Path(trace_dir)
        self.enabled = enabled
        self._lock = threading.Lock()
        if self.enabled:
            self.trace_dir.mkdir(parents=True, exist_ok=True)

    @property
    def trace_file(self) -> Path:
        return self.trace_dir / DEFAULT_TRACE_FILE

    def write(self, record: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with self._lock:
                with open(self.trace_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            # Never let tracing break the pipeline it is observing.
            pass

    def read_all(self) -> List[Dict[str, Any]]:
        """Read every span ever recorded. Used by the dashboard and CLI."""
        if not self.trace_file.exists():
            return []
        records = []
        try:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # A partially written line (process killed mid-write).
                        # Skip it rather than failing the whole read.
                        continue
        except Exception:
            return []
        return records

    def clear(self) -> None:
        try:
            if self.trace_file.exists():
                self.trace_file.unlink()
        except Exception:
            pass


_tracer: Optional[Tracer] = None


def get_tracer() -> Tracer:
    """Process-wide tracer. Created on first use."""
    global _tracer
    if _tracer is None:
        enabled = os.environ.get("TRACING_ENABLED", "1") != "0"
        _tracer = Tracer(enabled=enabled)
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Swap the tracer. Tests use this to write to a temp directory."""
    global _tracer
    _tracer = tracer


def _current_trace_id() -> Optional[str]:
    return getattr(_context, "trace_id", None)


def _current_span_id() -> Optional[str]:
    stack = getattr(_context, "span_stack", None)
    return stack[-1] if stack else None


@contextmanager
def trace(name: str, **metadata) -> Iterator[Dict[str, Any]]:
    """
    Start a new end-to-end trace. Everything traced inside this block belongs
    to the same request.

        with trace("query", user_id="rahim") as t:
            ...

    The yielded dict can be mutated to attach data discovered mid-request.
    """
    trace_id = uuid.uuid4().hex[:12]
    previous_trace = getattr(_context, "trace_id", None)
    previous_stack = getattr(_context, "span_stack", [])

    _context.trace_id = trace_id
    _context.span_stack = []

    payload: Dict[str, Any] = {"trace_id": trace_id, "metadata": dict(metadata)}
    start = time.perf_counter()
    error = None
    try:
        yield payload
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        get_tracer().write({
            "type": "trace",
            "trace_id": trace_id,
            "name": name,
            "timestamp": _now_iso(),
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "metadata": _truncate(payload.get("metadata", {})),
            "error": error,
        })
        _context.trace_id = previous_trace
        _context.span_stack = previous_stack


@contextmanager
def span(stage: str, **inputs) -> Iterator[Dict[str, Any]]:
    """
    Trace one stage of the pipeline.

        with span("retrieval", query=question) as s:
            chunks = search(question)
            s["outputs"]["chunk_count"] = len(chunks)

    Mutate s["outputs"] to record what the stage produced. Latency, nesting,
    and error capture are handled automatically.
    """
    span_id = uuid.uuid4().hex[:12]
    parent_id = _current_span_id()

    stack = getattr(_context, "span_stack", None)
    if stack is None:
        stack = []
        _context.span_stack = stack
    stack.append(span_id)

    record: Dict[str, Any] = {"inputs": dict(inputs), "outputs": {}}
    start = time.perf_counter()
    error = None
    try:
        yield record
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stack.pop()
        get_tracer().write({
            "type": "span",
            "trace_id": _current_trace_id(),
            "span_id": span_id,
            "parent_span_id": parent_id,
            "stage": stage,
            "timestamp": _now_iso(),
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
            "inputs": _truncate(record.get("inputs", {})),
            "outputs": _truncate(record.get("outputs", {})),
            "error": error,
        })
