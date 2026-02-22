"""Streamlit UI for the CA DMV RAG API. Calls the /ask endpoint.

Run from project root:
    streamlit run streamlit_app.py

Start the API first: uvicorn src.api.main:app --reload  (or ./scripts/run_api.sh)
Set API_URL in .env to override http://localhost:8000
"""

import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT_SECONDS = int(os.getenv("API_TIMEOUT_SECONDS", "180"))


def ask_api(
    question: str,
    stream: bool = False,
    include_sources: bool = True,
    source_filter: Optional[str] = None,
) -> tuple[str, list, str]:
    """Returns (answer, sources, confidence). If stream=True, uses /ask/stream."""
    if stream:
        return _ask_api_stream(question, include_sources=include_sources, source_filter=source_filter)
    url = f"{API_URL}/ask"
    body = {"question": question, "include_sources": include_sources}
    if source_filter:
        body["source_filter"] = source_filter
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as r:
            out = json.load(r)
            return (
                out.get("answer", ""),
                out.get("sources", []),
                out.get("confidence", "high"),
            )
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            d = json.loads(body) if body else {}
            if e.code == 422:
                errs = d.get("detail", [])
                if isinstance(errs, list) and errs:
                    msg = errs[0].get("msg", str(errs))
                else:
                    msg = "Question too long (max 1000 characters)." if len(question) > 1000 else str(d)
                raise RuntimeError(msg)
            detail = d.get("detail", body or str(e))
        except RuntimeError:
            raise
        except Exception:
            detail = body or str(e)
        raise RuntimeError(f"API error ({e.code}): {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach API at {API_URL}. Is it running? (e.g. uvicorn src.api.main:app --reload)"
        ) from e
    except TimeoutError as e:
        raise RuntimeError(f"API timeout after {API_TIMEOUT_SECONDS}s while calling /ask.") from e
    except socket.timeout as e:
        raise RuntimeError(f"API timeout after {API_TIMEOUT_SECONDS}s while calling /ask.") from e


def _ask_api_stream(
    question: str,
    include_sources: bool = True,
    source_filter: Optional[str] = None,
) -> tuple[str, list, str]:
    """Call POST /ask/stream, parse SSE, return (answer, sources, confidence)."""
    url = f"{API_URL}/ask/stream"
    body = {"question": question, "include_sources": include_sources}
    if source_filter:
        body["source_filter"] = source_filter
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as r:
            answer_parts = []
            sources = []
            confidence = "high"
            for line in r:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    if "token" in payload:
                        answer_parts.append(payload["token"])
                    if payload.get("done"):
                        sources = payload.get("sources", [])
                        confidence = payload.get("confidence", "high")
                        final_answer = payload.get("answer", "")
                        if final_answer:
                            return (final_answer, sources, confidence)
                        return ("".join(answer_parts), sources, confidence)
            return ("".join(answer_parts), sources, confidence)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            d = json.loads(body) if body else {}
            raise RuntimeError(d.get("detail", body or str(e)))
        except RuntimeError:
            raise
        except Exception:
            raise RuntimeError(body or str(e))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach API at {API_URL}. Is it running?"
        ) from e
    except TimeoutError as e:
        raise RuntimeError(f"API timeout after {API_TIMEOUT_SECONDS}s while calling /ask/stream.") from e
    except socket.timeout as e:
        raise RuntimeError(f"API timeout after {API_TIMEOUT_SECONDS}s while calling /ask/stream.") from e


def _stream_generator(
    question: str,
    result_holder: dict,
    include_sources: bool = True,
    source_filter: Optional[str] = None,
):
    """Generator that yields tokens from /ask/stream. Puts final answer/sources/confidence in result_holder."""
    url = f"{API_URL}/ask/stream"
    body = {"question": question, "include_sources": include_sources}
    if source_filter:
        body["source_filter"] = source_filter
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=API_TIMEOUT_SECONDS) as r:
            for line in r:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    try:
                        payload = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    if "error" in payload:
                        raise RuntimeError(payload["error"])
                    if "token" in payload:
                        yield payload["token"]
                    if payload.get("done"):
                        result_holder["answer"] = payload.get("answer", "")
                        result_holder["sources"] = payload.get("sources", [])
                        result_holder["confidence"] = payload.get("confidence", "high")
                        return
    except (TimeoutError, socket.timeout):
        raise RuntimeError(
            f"API stream timed out after {API_TIMEOUT_SECONDS}s. "
            "Try again or disable streaming in the sidebar."
        )


def main():
    st.set_page_config(page_title="CA DMV RAG", page_icon="📘", layout="centered")
    if "history" not in st.session_state:
        st.session_state.history = []  # list of {question, answer, sources, confidence}

    st.title("CA DMV Handbook Q&A")
    st.caption("Ask questions about the California DMV Driver Handbook. Answers are grounded in the handbook.")

    with st.sidebar:
        st.markdown("### API")
        st.text_input("API URL", value=API_URL, key="api_url", disabled=True)
        st.caption("Override with API_URL in .env")
        stream_answer = st.checkbox("Stream answer", value=True, help="Stream tokens as they arrive (uses /ask/stream)")
        include_sources = st.checkbox("Include sources", value=True, help="Show retrieved chunks (set False for answer only)")
        source_filter = st.text_input(
            "Source filter (optional)",
            value="",
            placeholder="e.g. ca-drivers-handbook",
            help="Only use chunks from this source when using multiple PDFs",
        ).strip() or None
        if st.session_state.history:
            if st.button("Clear history"):
                st.session_state.history.clear()
                st.rerun()

    # Show previous Q&As (oldest first)
    for item in st.session_state.history:
        with st.container():
            st.markdown("#### Q")
            st.markdown(item["question"])
            st.markdown("**A**")
            st.markdown(item["answer"])
            if item.get("sources"):
                with st.expander("Sources"):
                    for i, src in enumerate(item["sources"], 1):
                        page_label = f"Page {src.get('page')}" if src.get("page") is not None else "Page ?"
                        st.caption(f"Source {i}: {page_label} (score: {src.get('score', 0):.3f})")
                        st.text(src.get("snippet", ""))
            st.markdown("---")

    question = st.text_area(
        "Question",
        placeholder="e.g. What is the blood alcohol limit for DUI?",
        height=100,
        max_chars=1000,
        help="Max 1000 characters.",
    )
    if st.button("Ask"):
        if not (question or "").strip():
            st.warning("Enter a question.")
        elif len(question) > 1000:
            st.warning("Question is too long (max 1000 characters).")
        else:
            try:
                if stream_answer:
                    st.markdown("### Answer")
                    result_holder = {}
                    try:
                        stream_gen = _stream_generator(
                            question.strip(), result_holder,
                            include_sources=include_sources,
                            source_filter=source_filter,
                        )
                        answer_text = st.write_stream(stream_gen)
                        sources = result_holder.get("sources", [])
                        confidence = result_holder.get("confidence", "high")
                        if result_holder.get("answer"):
                            answer_text = result_holder["answer"]
                    except RuntimeError as e:
                        st.warning(f"{e} Falling back to non-stream mode for this request.")
                        with st.spinner("Retrying without streaming..."):
                            answer_text, sources, confidence = ask_api(
                                question.strip(), stream=False,
                                include_sources=include_sources,
                                source_filter=source_filter,
                            )
                    st.markdown("---")
                    if sources:
                        st.markdown("#### Sources (from handbook vector DB)")
                        for i, src in enumerate(sources, 1):
                            page_label = f"Page {src.get('page')}" if src.get("page") is not None else "Page ?"
                            with st.expander(f"Source {i}: {page_label} (score: {src.get('score', 0):.3f})"):
                                st.text(src.get("snippet", ""))
                else:
                    with st.spinner("Asking..."):
                        answer_text, sources, confidence = ask_api(
                            question.strip(), stream=False,
                            include_sources=include_sources,
                            source_filter=source_filter,
                        )
                    st.markdown("### Answer")
                    st.markdown(answer_text)
                    if sources:
                        st.markdown("---")
                        st.markdown("#### Sources (from handbook vector DB)")
                        for i, src in enumerate(sources, 1):
                            page_label = f"Page {src.get('page')}" if src.get("page") is not None else "Page ?"
                            with st.expander(f"Source {i}: {page_label} (score: {src.get('score', 0):.3f})"):
                                st.text(src.get("snippet", ""))
                st.session_state.history.append({
                    "question": question.strip(),
                    "answer": answer_text,
                    "sources": sources,
                    "confidence": confidence,
                })
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
