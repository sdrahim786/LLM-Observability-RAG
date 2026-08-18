"""
End-to-end demo. Ingests documents, asks questions, then prints the
observability report built from the traces those requests produced.

Runs with the real stack by default. Pass --fake to use the stand-in embedder
and LLM, which needs no model download and no Ollama:

    python scripts/demo.py --fake
    python scripts/demo.py            # requires ollama serve + llama3.1
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.observability import analytics
from src.observability.tracer import Tracer, set_tracer
from src.orchestration.assistant import Assistant


QUESTIONS = [
    "What chunk size should I use?",
    "What embedding model is this project using?",
    "What is the capital of France?",  # deliberately unanswerable from the docs
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", action="store_true", help="Use stand-in embedder and LLM")
    parser.add_argument("--data", default="data/raw", help="Folder to ingest")
    parser.add_argument("--keep", action="store_true", help="Keep the temp index and traces")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="llmobs_demo_"))
    set_tracer(Tracer(trace_dir=workdir / "traces"))

    assistant = Assistant(
        fake=args.fake,
        persist_dir=str(workdir / "chroma_db"),
        memory_path=str(workdir / "memory.json"),
    )

    if not args.fake and hasattr(assistant.llm, "is_available"):
        if not assistant.llm.is_available():
            print("Ollama is not reachable at http://localhost:11434")
            print("Start it with `ollama serve` and `ollama pull llama3.1`,")
            print("or re-run this script with --fake.")
            shutil.rmtree(workdir, ignore_errors=True)
            return 1

    print(f"\n=== Ingesting {args.data} ===")
    result = assistant.ingest_path(args.data)
    print(f"documents: {result['documents']}")
    print(f"chunks:    {result['chunks']}")
    print(f"indexed:   {result['collection_size']} vectors")

    print("\n=== Asking questions ===")
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        answer = assistant.ask(question)
        print(f"A: {answer['answer'][:300]}")
        print(f"   sources: {len(answer['sources'])}  "
              f"prompt: {answer['prompt_name']}  "
              f"tokens: {answer['tokens']['prompt']}+{answer['tokens']['completion']}")
        if not answer["sources"]:
            print("   (no relevant documents found -- this is the case that "
                  "produces made-up answers when it goes untraced)")

    print("\n=== Observability report ===")
    print(json.dumps(analytics.summary(), indent=2))

    if args.keep:
        print(f"\nTraces kept at: {workdir / 'traces' / 'traces.jsonl'}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
