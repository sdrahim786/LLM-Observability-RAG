"""
Streamlit frontend.

Two halves, matching the two halves of the project: a chat window for using the
assistant, and a dashboard for seeing what it actually did. The dashboard is
the more interesting half.

Run with:
    streamlit run frontend/app.py

Expects the API to be running:
    uvicorn src.backend.api:app --reload
"""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Local LLM Observability", layout="wide")


def api_get(path: str, **params):
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API request failed: {exc}")
        return None


def api_post(path: str, payload: dict):
    try:
        response = httpx.post(f"{API_URL}{path}", json=payload, timeout=180)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API request failed: {exc}")
        return None


# --- Sidebar: system state and ingestion ---

with st.sidebar:
    st.header("System")

    health = api_get("/health")
    if health:
        llm_up = health.get("llm_available")
        if llm_up is True:
            st.success(f"LLM: {health.get('model')}")
        elif llm_up is False:
            st.warning("Ollama not reachable. Start it with `ollama serve`.")
        else:
            st.info(f"LLM: {health.get('model')}")

        st.metric("Vectors indexed", health.get("vectors_indexed", 0))
        st.metric("Memories stored", health.get("memories_stored", 0))
        st.caption(f"Prompt version: {health.get('prompt_version')}")

    st.divider()
    st.header("Ingest documents")
    path = st.text_input("File or folder path", value="data/raw")
    if st.button("Ingest", use_container_width=True):
        with st.spinner("Loading, chunking, embedding..."):
            result = api_post("/ingest", {"path": path})
        if result:
            st.success(
                f"{result['documents']} document(s) to "
                f"{result['chunks']} chunk(s), indexed."
            )

chat_tab, dashboard_tab, traces_tab = st.tabs(["Chat", "Dashboard", "Traces"])


# --- Chat ---

with chat_tab:
    st.title("Ask your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"{len(message['sources'])} source(s)"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(
                            f"**[{i}] {source['source_file']}** "
                            f"(chunk {source['chunk_index']}, "
                            f"relevance {source['score']:.2f})"
                        )
                        st.caption(source["preview"])

    if question := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                result = api_post("/query", {"question": question, "remember": True})

            if result:
                st.markdown(result["answer"])

                if not result["sources"]:
                    st.warning(
                        "No relevant documents found. The answer above is the "
                        "model saying so, not an answer from your files."
                    )

                cols = st.columns(4)
                cols[0].metric("Chunks used", len(result["sources"]))
                cols[1].metric("Prompt tokens", result["tokens"]["prompt"])
                cols[2].metric("Completion tokens", result["tokens"]["completion"])
                cols[3].metric("Prompt variant", result["prompt_name"])
                st.caption(f"Trace ID: {result['trace_id']}")

                if result["sources"]:
                    with st.expander(f"{len(result['sources'])} source(s)"):
                        for i, source in enumerate(result["sources"], start=1):
                            st.markdown(
                                f"**[{i}] {source['source_file']}** "
                                f"(chunk {source['chunk_index']}, "
                                f"relevance {source['score']:.2f})"
                            )
                            st.caption(source["preview"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })


# --- Dashboard ---

with dashboard_tab:
    st.title("Pipeline observability")

    if st.button("Refresh"):
        st.rerun()

    metrics = api_get("/metrics")
    if metrics:
        cols = st.columns(4)
        cols[0].metric("Total requests", metrics["total_requests"])
        cols[1].metric("Avg request", f"{metrics['avg_request_ms']:.0f} ms")
        cols[2].metric("p95 request", f"{metrics['p95_request_ms']:.0f} ms")
        cols[3].metric("Total tokens", metrics["tokens"]["total_tokens"])

        st.subheader("Retrieval quality")
        retrieval = metrics["retrieval"]
        if retrieval["queries"]:
            cols = st.columns(4)
            cols[0].metric("Queries", retrieval["queries"])
            cols[1].metric(
                "Empty retrievals",
                retrieval["empty_retrievals"],
                delta=f"{retrieval['empty_rate']:.0%} rate",
                delta_color="inverse",
            )
            cols[2].metric("Avg chunks returned", retrieval["avg_chunks_returned"])
            if "avg_top_score" in retrieval:
                cols[3].metric("Avg top score", f"{retrieval['avg_top_score']:.3f}")

            if retrieval["empty_retrievals"]:
                st.warning(
                    f"{retrieval['empty_retrievals']} queries returned nothing. "
                    "These are the requests where the assistant had no grounding "
                    "and is most likely to disappoint."
                )
        else:
            st.info("No queries traced yet. Ask something in the Chat tab.")

        st.subheader("Where time goes")
        latency = metrics["latency_by_stage"]
        if latency:
            st.dataframe(
                [
                    {
                        "stage": stage,
                        "calls": data["calls"],
                        "avg ms": data["avg_ms"],
                        "p95 ms": data["p95_ms"],
                        "max ms": data["max_ms"],
                        "total ms": data["total_ms"],
                    }
                    for stage, data in latency.items()
                ],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "p95 matters more than the average. A stage that is usually "
                "fast but occasionally stalls hides in the mean."
            )
        else:
            st.info("No spans recorded yet.")

        st.subheader("Tokens")
        tokens = metrics["tokens"]
        if tokens["calls"]:
            cols = st.columns(4)
            cols[0].metric("LLM calls", tokens["calls"])
            cols[1].metric("Prompt tokens", tokens["prompt_tokens"])
            cols[2].metric("Completion tokens", tokens["completion_tokens"])
            cols[3].metric("Avg prompt tokens", tokens["avg_prompt_tokens"])
            st.caption(
                "Rising average prompt tokens usually means retrieval is "
                "returning more context than the question needs."
            )

        errors = metrics["errors"]
        if errors["failed_spans"]:
            st.subheader("Errors")
            st.error(
                f"{errors['failed_spans']} of {errors['total_spans']} spans failed "
                f"({errors['error_rate']:.1%})"
            )
            st.json(errors["by_stage"])


# --- Traces ---

with traces_tab:
    st.title("Recent traces")
    st.caption("One trace per request. Expand to see every stage it went through.")

    data = api_get("/traces", limit=20)
    if data:
        traces = data.get("traces", [])
        if not traces:
            st.info("No traces yet.")
        for t in traces:
            question = t.get("metadata", {}).get("question", t.get("name", "request"))
            label = f"{t.get('name')} | {t.get('duration_ms')} ms | {str(question)[:60]}"
            with st.expander(label):
                st.caption(f"Trace ID: {t.get('trace_id')}  |  {t.get('timestamp')}")
                if t.get("error"):
                    st.error(t["error"])
                st.json(t.get("metadata", {}))
                for s in t.get("spans", []):
                    marker = "FAILED " if s.get("error") else ""
                    st.markdown(f"**{marker}{s['stage']}** | {s['duration_ms']} ms")
                    if s.get("outputs"):
                        st.json(s["outputs"])
                    if s.get("error"):
                        st.error(s["error"])
