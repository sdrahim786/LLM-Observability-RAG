"""
LLM client -- Llama 3.1 through Ollama, running locally.

Ollama's HTTP API is called directly rather than through a wrapper. The reason
is observability: the raw response carries token counts and timing breakdowns
(how long the model spent loading vs. evaluating the prompt vs. generating)
that most wrappers discard, and those are exactly the numbers worth tracing.

FakeLLM exists so the pipeline, prompts, and tracing can be tested without a
running Ollama instance.
"""

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from ..observability.tracer import span

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_TIMEOUT = 120


@runtime_checkable
class LLM(Protocol):
    model: str

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]: ...


class OllamaLLM:
    """
    Talks to a local Ollama server.

    Expects `ollama serve` to be running and the model pulled:
        ollama pull llama3.1
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.2,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature

    def is_available(self) -> bool:
        """Check the server is up before trying to use it."""
        try:
            import httpx

            response = httpx.get(f"{self.host}/api/tags", timeout=3)
            return response.status_code == 200
        except Exception:
            return False

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        with span("llm", model=self.model, prompt_chars=len(prompt)) as s:
            try:
                import httpx
            except ImportError as exc:
                raise ImportError("httpx is required. Run: pip install httpx") from exc

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": kwargs.get("temperature", self.temperature)},
            }

            response = httpx.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            text = data.get("response", "")

            # Ollama reports these directly. prompt_eval_count is how many
            # tokens the context cost, which is where retrieved chunks show up
            # -- if this number balloons, retrieval is returning too much.
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            s["outputs"]["prompt_tokens"] = prompt_tokens
            s["outputs"]["completion_tokens"] = completion_tokens
            s["outputs"]["total_tokens"] = prompt_tokens + completion_tokens
            s["outputs"]["response_chars"] = len(text)
            # Nanoseconds from Ollama, converted for readability.
            if data.get("eval_duration"):
                s["outputs"]["generation_ms"] = round(data["eval_duration"] / 1e6, 2)
            if data.get("load_duration"):
                s["outputs"]["model_load_ms"] = round(data["load_duration"] / 1e6, 2)

            return {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model,
            }


class FakeLLM:
    """
    Deterministic stand-in. Echoes back a summary of what it was given so
    tests can assert on prompt construction, and estimates token counts by
    the usual rough heuristic of ~4 characters per token.
    """

    def __init__(self, model: str = "fake", canned_response: Optional[str] = None):
        self.model = model
        self.canned_response = canned_response

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        with span("llm", model=self.model, prompt_chars=len(prompt)) as s:
            text = self.canned_response or (
                "This is a test response generated without a real model."
            )
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(text) // 4)

            s["outputs"]["prompt_tokens"] = prompt_tokens
            s["outputs"]["completion_tokens"] = completion_tokens
            s["outputs"]["total_tokens"] = prompt_tokens + completion_tokens
            s["outputs"]["response_chars"] = len(text)

            return {
                "text": text,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": self.model,
            }


def get_llm(fake: bool = False, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> LLM:
    if fake:
        return FakeLLM()
    return OllamaLLM(model=model, host=host)
