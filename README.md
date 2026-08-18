<<<<<<< HEAD
# Local LLM Observability

Trace every LLM call, retrieval, and memory operation in a local RAG pipeline. Fully self-hosted, no paid APIs, no data leaving your machine.

## Why

Local LLM pipelines are easy to build and hard to see inside. A slow embedding call, a bad retrieval, a chunk that never should have matched: none of it surfaces unless you log it yourself.

Most reports of "the model hallucinated" turn out to be something more specific. Retrieval returned nothing, so the model answered from its own priors. Or it returned five chunks of loosely related text and the model tried to use all of them. Without tracing there is no way to tell those cases apart, and no way to know which one you are fixing.

This project instruments a complete RAG pipeline so that every request answers one question: what actually happened on that call?

## What gets traced

Every request produces one trace containing a span per stage:

| Stage | Recorded |
|---|---|
| ingestion | documents loaded, total characters |
| chunking | chunk count, size distribution, per-source counts |
| embedding | vectors produced, dimensions, batch size |
| vectorstore | writes, collection growth, search latency |
| retrieval | chunks returned, similarity scores, **empty-retrieval flag**, sources |
| memory | reads and writes, hit counts, kept separate from document retrieval |
| prompt_build | which prompt variant fired, prompt size, context and memories used |
| llm | prompt/completion tokens, generation time, model load time |

Spans nest, so a trace forms a tree and you can see where time actually went. Errors are captured with the span that failed.

## Quick start

```bash
git clone https://github.com/<your-username>/llm-observability-local.git
cd llm-observability-local

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full pipeline with stand-in backends (no model download, no Ollama):

```bash
python scripts/demo.py --fake
```

That ingests the sample documents, asks three questions including one the documents cannot answer, and prints the observability report built from the resulting traces:

```
Q: What is the capital of France?
   sources: 0  prompt: no_context  tokens: 147+13
   (no relevant documents found)

"retrieval": {
  "queries": 3,
  "empty_retrievals": 1,
  "empty_rate": 0.333,
  "avg_top_score": 0.2842
}
```

That `empty_rate` is the headline number. It is the share of questions where the assistant had nothing to ground its answer in.

## Running for real

Requires [Ollama](https://ollama.com):

```bash
ollama serve
ollama pull llama3.1
```

Then:

```bash
python scripts/demo.py                              # full stack, real model
uvicorn src.backend.api:app --reload                # API on :8000
streamlit run frontend/app.py                       # UI on :8501
```

The Streamlit app has three tabs: chat with your documents, a dashboard of pipeline metrics, and a trace viewer showing every stage of recent requests.

## API

Assistant:

- `POST /ingest` load, chunk, embed and index a file or folder
- `POST /query` ask a question, get an answer with sources and token counts
- `GET/POST/DELETE /memories` inspect and manage long-term memory

Observability:

- `GET /metrics` everything below, in one call
- `GET /metrics/latency` per-stage timing including p95
- `GET /metrics/retrieval` empty-retrieval rate, score distribution
- `GET /metrics/tokens` token usage across LLM calls
- `GET /metrics/errors` failure counts grouped by stage
- `GET /traces` recent requests with their full span trees

## Layout

```
src/
  ingestion/       loaders for pdf, txt, md, docx
  chunking/        recursive splitting with chunk stats
  embeddings/      all-MiniLM-L6-v2 locally, plus a deterministic fake
  vectorstore/     ChromaDB on disk, cosine similarity
  retrieval/       search with score filtering and empty detection
  memory/          long-term memory, JSON-backed by default, mem0 optional
  orchestration/   versioned prompts, Ollama client, the assembled assistant
  observability/   tracer and analytics -- the core of the project
  backend/         FastAPI
frontend/          Streamlit chat + dashboard + trace viewer
scripts/           runnable demos
tests/             50 tests, no network required
data/raw/          drop your documents here
```

## Design notes

**Tracing never breaks the pipeline.** Every write path in `tracer.py` swallows its own errors. Observability that can crash the thing it observes is worse than none.

**Every backend is swappable.** The embedder and LLM are injected, not constructed internally, and both have deterministic stand-ins. That is why the whole test suite runs offline in under three seconds.

**Memory is not the vector store.** Documents and facts about the user live in separate stores with separate traces. Conflating them is the most common bug in memory-augmented RAG.

**Prompts are versioned.** The prompt version is written into every trace, so when answer quality shifts you can tell whether it was a prompt edit, a retrieval change, or a model change.

**Re-ingesting is idempotent.** Chunk IDs are derived from source file and chunk index, so running ingest twice overwrites rather than silently doubling your index.

**Weak matches are dropped, not passed along.** Chunks below the similarity threshold are filtered out rather than handed to the model, because irrelevant context actively degrades answers.

## Tests

```bash
pytest tests/ -q
```

50 tests covering the tracer, analytics, every pipeline stage, the assembled assistant, and the API. They use the stand-in embedder and LLM, so no model download or running Ollama is needed.

## Known limitations

`LocalMemory` scores recall by keyword overlap, not semantically, so it misses paraphrases. Install `mem0ai` and switch backends for semantic recall.

Traces are append-only JSONL with no rotation. Fine for personal use; add rotation before pointing this at anything long-running.

Scanned PDFs contain no extractable text and will ingest as empty. OCR is out of scope.

## License

MIT
=======
# LLM-Observability-RAG
Open source tracing and observability for local LLM apps. Logs latency, token cost, and retrieval quality for every call, built entirely on self-hosted tools
>>>>>>> 4b30322d00a2d8286a84c149b9e3341f14194cf8
