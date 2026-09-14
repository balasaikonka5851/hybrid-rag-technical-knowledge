"""
HybridRAG — Strong Final Streamlit UI
======================================

Production-style presentation layer for the existing HybridRAG backend.

Highlights
----------
- Responsive desktop/mobile layout using native Streamlit components.
- Project-relative architecture image.
- Evidence-first answer presentation.
- Citation validation.
- Explicit Gemini quota/rate-limit handling.
- No automatic retry storms that can consume more quota.
- Session-level successful-answer cache to avoid duplicate generation calls.
- Safe "Retry last query" control after transient/provider failures.
- Clear provider/backend status.
- Retrieval diagnostics only use fields actually returned by RAGPipeline.
- No secrets, API keys, stack traces, or raw provider responses are displayed.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import streamlit as st


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "app" / "assets"
ARCHITECTURE_IMAGE = ASSETS_DIR / "hybridrag_architecture.png"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# BACKEND IMPORTS
# ============================================================================

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.citation_validator import validate_citations


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="HybridRAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CONSTANTS
# ============================================================================

APP_VERSION = "2.0"
MAX_QUERY_LENGTH = 2000
CACHE_LIMIT = 20

EXAMPLES = [
    "How do I create dependencies in FastAPI?",
    "How does Python handle exceptions?",
    "How can I validate data received by an API?",
    "What is RequestValidationError in FastAPI?",
    "How does Python's yield keyword work?",
]


# ============================================================================
# SESSION STATE
# ============================================================================

DEFAULT_STATE = {
    "last_result": None,
    "last_query": "",
    "last_status": "idle",
    "answer_cache": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# SAFE HELPERS
# ============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def ms(value: Any) -> str:
    return f"{safe_float(value):,.0f} ms"


def seconds(value: Any) -> str:
    return f"{safe_float(value) / 1000:.1f} s"


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def query_key(query: str) -> str:
    normalized = normalize_query(query).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_location(source: dict[str, Any]) -> str:
    parts: list[str] = []

    if source.get("page") is not None:
        parts.append(f"Page {source['page']}")

    if source.get("sheet"):
        parts.append(f"Sheet: {source['sheet']}")

    if source.get("slide") is not None:
        parts.append(f"Slide {source['slide']}")

    return " · ".join(parts)


def is_quota_error(result: dict[str, Any]) -> bool:
    error_type = str(result.get("error_type") or "").lower()
    error_message = str(result.get("error_message") or "").lower()

    indicators = (
        "429",
        "quota",
        "resource_exhausted",
        "too_many_requests",
        "rate limit",
        "rate_limit",
        "generate_content_free_tier_requests",
    )

    return any(item in error_type or item in error_message for item in indicators)


def is_retrieval_error(result: dict[str, Any]) -> bool:
    error_type = str(result.get("error_type") or "").lower()
    return "retrieval" in error_type


def is_generation_error(result: dict[str, Any]) -> bool:
    error_type = str(result.get("error_type") or "").lower()
    return "generation" in error_type or "api" in error_type


def render_metrics(items: list[tuple[str, str, str]]) -> None:
    columns = st.columns(len(items), gap="small")

    for column, (label, value, help_text) in zip(columns, items):
        with column:
            st.metric(label, value, help=help_text)


def cache_successful_result(query: str, result: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        return

    if not result.get("success", False):
        return

    cache = st.session_state.answer_cache
    cache[query_key(query)] = {
        "query": normalize_query(query),
        "result": result,
    }

    # Keep only the newest CACHE_LIMIT entries.
    while len(cache) > CACHE_LIMIT:
        oldest_key = next(iter(cache))
        del cache[oldest_key]


def get_cached_result(query: str) -> dict[str, Any] | None:
    item = st.session_state.answer_cache.get(query_key(query))
    if not isinstance(item, dict):
        return None

    result = item.get("result")
    return result if isinstance(result, dict) else None


def clear_current_result() -> None:
    st.session_state.last_result = None
    st.session_state.last_query = ""
    st.session_state.last_status = "idle"


# ============================================================================
# LIGHTWEIGHT RESPONSIVE CSS
# ============================================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        padding-left: clamp(0.75rem, 3vw, 3rem);
        padding-right: clamp(0.75rem, 3vw, 3rem);
    }

    .app-subtitle {
        font-size: 1rem;
        line-height: 1.55;
        opacity: 0.72;
        max-width: 920px;
        margin-top: -0.35rem;
        margin-bottom: 0.7rem;
    }

    .trust-line {
        font-size: 0.82rem;
        line-height: 1.8;
        opacity: 0.70;
        margin-bottom: 1.1rem;
    }

    .small-note {
        font-size: 0.78rem;
        opacity: 0.65;
    }

    div[data-testid="stMetric"] {
        min-height: 86px;
    }

    @media (max-width: 600px) {
        .block-container {
            padding-top: 0.65rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .app-subtitle {
            font-size: 0.9rem;
        }

        .trust-line {
            font-size: 0.74rem;
        }

        div[data-testid="stMetric"] {
            min-height: 76px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HEADER
# ============================================================================

st.title("🔎 HybridRAG")

st.markdown(
    """
    <div class="app-subtitle">
    Intelligent technical knowledge retrieval with hybrid search,
    evidence-grounded generation, adaptive fusion, and citation validation.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="trust-line">
    🧠 Semantic Search &nbsp;•&nbsp;
    🔤 BM25 &nbsp;•&nbsp;
    🔀 Weighted RRF &nbsp;•&nbsp;
    📚 Evidence First &nbsp;•&nbsp;
    🔗 Citation Validation &nbsp;•&nbsp;
    🛡️ Grounded Answers
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# PIPELINE INITIALIZATION
# ============================================================================

@st.cache_resource(show_spinner="Initializing HybridRAG…")
def get_pipeline() -> RAGPipeline:
    """Initialize the expensive retrieval/generation stack once."""
    return RAGPipeline()


pipeline_error = None

try:
    pipeline = get_pipeline()
except Exception as exc:
    pipeline = None
    pipeline_error = exc


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ System")

    if pipeline is not None:
        st.success("HybridRAG ready")
    else:
        st.error("Backend unavailable")

    st.subheader("Knowledge Base")
    st.write("**Corpus:** Python + FastAPI")
    st.write("**Documents:** 696")
    st.write("**Chunks:** 18,150")
    st.write("**Embeddings:** MiniLM · 384-D")
    st.write("**Indexes:** FAISS + BM25")

    st.subheader("Retrieval")
    st.write("**Semantic candidates:** 60")
    st.write("**BM25 candidates:** 60")
    st.write("**Base weights:** 0.75 / 0.25")
    st.write("**Fusion:** Weighted RRF · k=60")
    st.write("**Adaptive fusion:** ON")

    st.subheader("Generation")
    st.write("**Model:** Gemini 3.6 Flash")
    st.write("**Context:** 5 chunks")

    st.subheader("Guardrails")
    st.write("✓ Query validation")
    st.write("✓ Evidence preservation")
    st.write("✓ Citation validation")
    st.write("✓ Groundedness evaluation")
    st.write("✓ Provider failure handling")
    st.write("✓ Duplicate-query cache")

    st.divider()

    st.subheader("Provider resilience")

    last_result = st.session_state.last_result

    if isinstance(last_result, dict) and not last_result.get("success", True):
        if is_quota_error(last_result):
            st.warning("Gemini quota/rate limit detected")
            st.caption(
                "No automatic retry loop is used. Repeated retries can "
                "consume additional request quota."
            )
        elif is_retrieval_error(last_result):
            st.error("Retrieval failure")
        elif is_generation_error(last_result):
            st.warning("Generation failure")
        else:
            st.warning("Pipeline failure")
    else:
        st.info("Provider status is checked when a query is executed.")

    st.divider()

    if st.button(
        "🧹 Clear current answer",
        use_container_width=True,
        disabled=st.session_state.last_result is None,
    ):
        clear_current_result()
        st.rerun()

    if st.button(
        "🗑️ Clear answer cache",
        use_container_width=True,
        disabled=not bool(st.session_state.answer_cache),
    ):
        st.session_state.answer_cache.clear()
        st.toast("Answer cache cleared.")
        st.rerun()

    st.caption(
        "Ask about APIs, concepts, parameters, exceptions, behavior, "
        "or implementation details."
    )
    st.caption(f"UI v{APP_VERSION}")


# ============================================================================
# BACKEND FAILURE
# ============================================================================

if pipeline is None:
    st.error(
        "HybridRAG could not initialize the backend. "
        "Verify the knowledge-base artifact, dependencies, and environment "
        "configuration."
    )

    with st.expander("Initialization diagnostics"):
        st.write(f"Exception type: `{type(pipeline_error).__name__}`")
        st.write(
            "Detailed exception text is intentionally hidden from the public "
            "interface."
        )

    st.stop()


# ============================================================================
# ABOUT + ARCHITECTURE
# ============================================================================

with st.expander("📘 About HybridRAG", expanded=True):
    st.markdown("### Technical Knowledge Retrieval & Question Answering")

    st.write(
        "HybridRAG combines semantic vector retrieval and BM25 keyword "
        "retrieval to find relevant technical evidence. Adaptive weighted "
        "fusion combines the candidate lists. Selected evidence is passed "
        "to the language model, and generated citations are checked against "
        "the returned sources."
    )

    render_metrics(
        [
            ("Retrieval", "Hybrid", "Semantic + BM25"),
            ("Fusion", "RRF", "Weighted Reciprocal Rank Fusion"),
            ("Generation", "Grounded", "Evidence-first context"),
            ("Trust", "Validated", "Citation checking"),
        ]
    )

    st.markdown("### 🏗️ System Architecture")

    if ARCHITECTURE_IMAGE.is_file():
        st.image(
            str(ARCHITECTURE_IMAGE),
            use_container_width=True,
            caption=(
                "HybridRAG: retrieve → fuse → select evidence → "
                "generate → validate → answer."
            ),
        )
    else:
        st.warning(
            "Architecture image not found. Expected: "
            "`app/assets/hybridrag_architecture.png`"
        )


# ============================================================================
# KNOWLEDGE BASE DASHBOARD
# ============================================================================

st.header("📊 Knowledge Base")
st.caption(
    "Indexed technical documentation and retrieval configuration."
)

render_metrics(
    [
        ("Documents", "696", "Logical documents"),
        ("Chunks", "18,150", "Indexed retrieval units"),
        ("Embedding", "384-D", "MiniLM vector dimension"),
        ("Technologies", "2", "Python + FastAPI"),
    ]
)

render_metrics(
    [
        ("Python Docs", "536", "Indexed Python documents"),
        ("FastAPI Docs", "160", "Indexed FastAPI documents"),
        ("Chunk Size", "1,200", "Maximum characters"),
        ("Overlap", "200", "Characters"),
    ]
)


# ============================================================================
# RETRIEVAL EVALUATION
# ============================================================================

with st.expander("📈 Retrieval Evaluation", expanded=False):
    st.write(
        "40 technical queries were evaluated using manually audited "
        "relevant sources."
    )

    st.markdown(
        """
        | Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
        |---|---:|---:|---:|---:|---:|
        | BM25 | 0.175 | 0.300 | 0.425 | 0.550 | 0.271 |
        | Semantic | 0.350 | 0.475 | 0.575 | 0.700 | 0.442 |
        | Hybrid | 0.325 | 0.700 | 0.700 | **0.875** | **0.489** |
        """
    )

    st.success(
        "Best measured hybrid configuration: 60 candidates per retriever, "
        "87.5% Hit@10 and 0.489 MRR."
    )

    st.caption("MRR = Mean Reciprocal Rank")


# ============================================================================
# EXPERIMENTS
# ============================================================================

with st.expander("🧪 Retrieval Experiments", expanded=False):
    st.markdown("### Hybrid Weight Experiment")

    st.markdown(
        """
        | Semantic | BM25 | MRR |
        |---:|---:|---:|
        | 1.00 | 0.00 | 0.442 |
        | 0.75 | 0.25 | **0.442** |
        | 0.50 | 0.50 | 0.429 |
        | 0.25 | 0.75 | 0.375 |
        | 0.00 | 1.00 | 0.271 |
        """
    )

    st.markdown("### Reranking Experiment")

    st.write(
        "A Cross-Encoder MS-MARCO reranker was evaluated. It improved "
        "Hit@1 slightly but reduced deeper retrieval recall and overall "
        "MRR. It therefore remains an experiment rather than the primary "
        "ranking stage."
    )

    st.warning(
        "Primary retrieval uses hybrid retrieval without the cross-encoder reranker."
    )


# ============================================================================
# SEARCH
# ============================================================================

st.header("💬 Ask the Technical Knowledge Base")

st.caption(
    "Run the complete HybridRAG retrieval → generation → validation pipeline."
)

with st.expander("ℹ️ How a search is processed", expanded=False):
    st.markdown(
        """
        **1.** Normalize and validate the question.  
        **2.** Run semantic retrieval with MiniLM + FAISS.  
        **3.** Run BM25 keyword retrieval.  
        **4.** Apply adaptive weighted Reciprocal Rank Fusion.  
        **5.** Select the best evidence chunks.  
        **6.** Generate only from the selected evidence.  
        **7.** Validate citations against returned sources.  
        **8.** If Gemini is rate-limited, do **not** display an invented answer.
        """
    )

selected_example = st.selectbox(
    "Quick start",
    ["Custom question", *EXAMPLES],
)

default_query = "" if selected_example == "Custom question" else selected_example

query = st.text_area(
    "Technical question",
    value=default_query,
    height=120,
    max_chars=MAX_QUERY_LENGTH,
    placeholder="Example: How do dependencies work in FastAPI?",
    help=(
        "Ask about a concept, API, parameter, exception, behavior, "
        "or implementation detail."
    ),
)

c1, c2 = st.columns(2)

with c1:
    st.caption("💡 Focused questions generally produce clearer evidence.")

with c2:
    st.caption("🔎 Semantic + keyword retrieval is used together.")


ask_clicked = st.button(
    "🚀 Ask HybridRAG",
    type="primary",
    use_container_width=True,
)


# ============================================================================
# EXECUTION
# ============================================================================

if ask_clicked:
    cleaned_query = normalize_query(query)

    if not cleaned_query:
        st.warning("Please enter a technical question.")
        st.stop()

    if len(cleaned_query) < 3:
        st.warning("Please provide a more descriptive question.")
        st.stop()

    if len(cleaned_query) > MAX_QUERY_LENGTH:
        st.warning(
            f"Please keep the question under {MAX_QUERY_LENGTH:,} characters."
        )
        st.stop()

    # ------------------------------------------------------------
    # SESSION CACHE
    # ------------------------------------------------------------
    cached = get_cached_result(cleaned_query)

    if cached is not None:
        st.session_state.last_result = cached
        st.session_state.last_query = cleaned_query
        st.session_state.last_status = "cached"
        st.toast("Loaded the successful answer from the session cache.")
        st.rerun()

    # ------------------------------------------------------------
    # LIVE PIPELINE
    # ------------------------------------------------------------
    try:
        with st.spinner(
            "🔎 Retrieving evidence → selecting context → generating answer…"
        ):
            result = pipeline.ask(cleaned_query)

        if not isinstance(result, dict):
            st.error("The backend returned an unexpected response.")
            st.stop()

        st.session_state.last_result = result
        st.session_state.last_query = cleaned_query

        if result.get("success", False):
            st.session_state.last_status = "success"
            cache_successful_result(cleaned_query, result)
        elif is_quota_error(result):
            st.session_state.last_status = "quota"
        elif is_retrieval_error(result):
            st.session_state.last_status = "retrieval_error"
        else:
            st.session_state.last_status = "error"

    except Exception:
        st.session_state.last_status = "exception"
        st.error(
            "The request could not be completed. "
            "No unreliable answer was displayed."
        )
        st.stop()


# ============================================================================
# RETRY LAST QUERY
# ============================================================================

result = st.session_state.last_result

if isinstance(result, dict) and not result.get("success", True):
    if st.session_state.last_query:
        st.divider()
        st.subheader("🔁 Recovery")

        if is_quota_error(result):
            st.warning(
                "Gemini is currently rate-limited or out of quota. "
                "Automatic retry loops are intentionally disabled."
            )

            st.caption(
                "If you have upgraded the Gemini project or the quota has "
                "reset, retry this exact question."
            )

        elif is_retrieval_error(result):
            st.warning(
                "The retrieval stage failed. A single manual retry is available."
            )

        elif is_generation_error(result):
            st.warning(
                "The generation stage failed. A single manual retry is available."
            )

        retry_col, cache_col = st.columns(2)

        with retry_col:
            retry_clicked = st.button(
                "🔄 Retry last query",
                type="secondary",
                use_container_width=True,
            )

        with cache_col:
            cached_previous = get_cached_result(st.session_state.last_query)

            use_cache_clicked = st.button(
                "📦 Use cached answer",
                use_container_width=True,
                disabled=cached_previous is None,
            )

        if use_cache_clicked and cached_previous is not None:
            st.session_state.last_result = cached_previous
            st.session_state.last_status = "cached"
            st.rerun()

        if retry_clicked:
            retry_query = st.session_state.last_query

            try:
                with st.spinner("🔄 Retrying HybridRAG once…"):
                    retry_result = pipeline.ask(retry_query)

                if isinstance(retry_result, dict):
                    st.session_state.last_result = retry_result

                    if retry_result.get("success", False):
                        st.session_state.last_status = "success"
                        cache_successful_result(
                            retry_query,
                            retry_result,
                        )
                    elif is_quota_error(retry_result):
                        st.session_state.last_status = "quota"
                    else:
                        st.session_state.last_status = "error"

                    st.rerun()

                st.error("The backend returned an unexpected retry response.")

            except Exception:
                st.error(
                    "The retry failed. No unreliable answer was displayed."
                )


# ============================================================================
# RESULT
# ============================================================================

result = st.session_state.last_result

if isinstance(result, dict):
    st.divider()

    st.subheader("🔍 Question")
    st.info(st.session_state.last_query)

    if st.session_state.last_status == "cached":
        st.caption("📦 Served from the current Streamlit session cache.")

    success = bool(result.get("success", True))
    answerable = bool(result.get("answerable", True))

    # ========================================================================
    # FAILURE STATE
    # ========================================================================

    if not success:
        error_type = str(result.get("error_type") or "").lower()

        if is_quota_error(result):
            st.error("⚠️ Gemini generation is temporarily unavailable")

            st.info(
                "The retrieval pipeline reached the language-model generation "
                "stage, but Gemini rejected the generation request because the "
                "project is rate-limited or its current quota is exhausted."
            )

            st.markdown(
                """
                **What this means**

                - Your HybridRAG retrieval/indexing system is separate from Gemini.
                - The UI will not fabricate an answer when generation fails.
                - Automatic retry loops are disabled to avoid repeatedly consuming quota.
                - After the quota becomes available or billing/rate limits are upgraded,
                  use **Retry last query** above.
                """
            )

            st.caption(
                "This is a provider quota/rate-limit condition, not evidence that "
                "the retrieved documentation is incorrect."
            )

        elif is_retrieval_error(result):
            st.error("⚠️ Retrieval could not be completed")

            st.info(
                "HybridRAG could not retrieve the required evidence from the "
                "knowledge base. No generated answer was displayed."
            )

        elif is_generation_error(result):
            st.error("⚠️ Answer generation could not be completed")

            st.info(
                "The language-model generation stage failed. "
                "No unsupported answer was displayed."
            )

        else:
            st.error("⚠️ HybridRAG could not complete this request")

            st.info(
                "The pipeline encountered an internal problem. "
                "No unreliable answer was displayed."
            )

        st.subheader("🛡️ Safety behavior")

        render_metrics(
            [
                ("Answer", "Blocked", "No unreliable answer displayed"),
                ("Provider", "Failed", "Generation provider did not respond successfully"),
                ("Guardrail", "Active", "Failure propagated safely"),
            ]
        )

    # ========================================================================
    # NOT ANSWERABLE
    # ========================================================================

    elif not answerable:
        st.warning(
            "The knowledge base did not contain enough relevant evidence "
            "to answer this question reliably."
        )

        st.info(
            "Try a more specific question about Python or FastAPI, "
            "for example a named API, parameter, exception, or behavior."
        )

    # ========================================================================
    # SUCCESS
    # ========================================================================

    else:
        answer = str(result.get("answer") or "")
        sources = result.get("sources", [])

        if not isinstance(sources, list):
            sources = []

        # --------------------------------------------------------------------
        # CITATION VALIDATION
        # --------------------------------------------------------------------

        try:
            citation_result = validate_citations(
                answer=answer,
                sources=sources,
            )
        except Exception:
            citation_result = {
                "has_citations": False,
                "all_citations_valid": False,
                "valid_citations": [],
                "invalid_citations": [],
            }

        valid_citations = citation_result.get(
            "valid_citations",
            [],
        )

        invalid_citations = citation_result.get(
            "invalid_citations",
            [],
        )

        if not isinstance(valid_citations, list):
            valid_citations = []

        if not isinstance(invalid_citations, list):
            invalid_citations = []

        all_valid = bool(
            citation_result.get(
                "all_citations_valid",
                False,
            )
        )

        has_citations = bool(
            citation_result.get(
                "has_citations",
                False,
            )
        )

        # --------------------------------------------------------------------
        # ANSWER QUALITY
        # --------------------------------------------------------------------

        st.subheader("🛡️ Answer Quality")

        render_metrics(
            [
                (
                    "Evidence",
                    str(
                        safe_int(
                            result.get("context_chunks"),
                            len(sources),
                        )
                    ),
                    "Selected context chunks",
                ),
                (
                    "Valid Citations",
                    str(len(valid_citations)),
                    "Citations matched to evidence",
                ),
                (
                    "Invalid Citations",
                    str(len(invalid_citations)),
                    "Citations not matched to evidence",
                ),
                (
                    "Candidates",
                    str(
                        safe_int(
                            result.get("retrieved_candidates"),
                            0,
                        )
                    ),
                    "Hybrid retrieval candidates",
                ),
            ]
        )

        if all_valid:
            st.success(
                "✓ All generated citations map to returned evidence."
            )
        elif has_citations:
            st.warning(
                "Some generated citations could not be matched to returned evidence."
            )
        else:
            st.warning(
                "No source citations were detected in the generated answer."
            )

        # --------------------------------------------------------------------
        # ANSWER
        # --------------------------------------------------------------------

        st.subheader("🧠 Answer")

        if answer:
            st.markdown(answer)
        else:
            st.warning("No answer was generated.")

        # --------------------------------------------------------------------
        # PERFORMANCE
        # --------------------------------------------------------------------

        st.subheader("⚡ Performance")

        render_metrics(
            [
                (
                    "Retrieval",
                    ms(result.get("retrieval_latency_ms")),
                    "Search + fusion",
                ),
                (
                    "Generation",
                    seconds(result.get("generation_latency_ms")),
                    "LLM generation",
                ),
                (
                    "Total",
                    seconds(result.get("total_latency_ms")),
                    "End-to-end pipeline",
                ),
                (
                    "Sources",
                    str(len(sources)),
                    "Returned evidence sources",
                ),
            ]
        )

        # --------------------------------------------------------------------
        # DETAILS
        # --------------------------------------------------------------------

        tab_sources, tab_diagnostics, tab_raw = st.tabs(
            [
                "📚 Evidence & Sources",
                "🔬 Retrieval Diagnostics",
                "🧩 Raw Result",
            ]
        )

        # ====================================================================
        # EVIDENCE
        # ====================================================================

        with tab_sources:
            if not sources:
                st.info("No source metadata was returned.")
            else:
                for index, source in enumerate(
                    sources,
                    start=1,
                ):
                    if not isinstance(source, dict):
                        continue

                    source_name = source.get(
                        "source",
                        "unknown",
                    )

                    technology = source.get(
                        "technology",
                        "unknown",
                    )

                    section = source.get(
                        "section",
                        "General",
                    )

                    score = safe_float(
                        source.get(
                            "score",
                            source.get(
                                "rrf_score",
                                0,
                            ),
                        )
                    )

                    location = source_location(source)

                    with st.expander(
                        f"[SOURCE {index}] {source_name}",
                        expanded=(index == 1),
                    ):
                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.write("**Technology**")
                            st.write(str(technology))

                        with c2:
                            st.write("**Section**")
                            st.write(str(section))

                        with c3:
                            st.write("**Relevance**")
                            st.write(f"{score:.6f}")

                        if location:
                            st.caption(location)

                        source_text = source.get(
                            "text",
                            "",
                        )

                        if source_text:
                            st.markdown("**Retrieved evidence**")
                            st.code(
                                str(source_text),
                                language="text",
                            )

        # ====================================================================
        # RETRIEVAL DIAGNOSTICS
        # ====================================================================

        with tab_diagnostics:
            semantic_count = safe_int(
                result.get("semantic_candidates"),
                0,
            )

            bm25_count = safe_int(
                result.get("bm25_candidates"),
                0,
            )

            hybrid_count = safe_int(
                result.get("retrieved_candidates"),
                0,
            )

            context_count = safe_int(
                result.get("context_chunks"),
                len(sources),
            )

            render_metrics(
                [
                    (
                        "Semantic",
                        str(semantic_count),
                        "Semantic candidate count",
                    ),
                    (
                        "BM25",
                        str(bm25_count),
                        "Keyword candidate count",
                    ),
                    (
                        "Hybrid",
                        str(hybrid_count),
                        "Fused candidate count",
                    ),
                    (
                        "Context",
                        str(context_count),
                        "Selected evidence chunks",
                    ),
                ]
            )

            st.markdown("#### Retrieval flow")

            st.code(
                """User Query
     │
     ├── Semantic Search → MiniLM → FAISS
     │
     └── BM25 Keyword Search
              │
              ▼
       Adaptive Weighted RRF
              │
              ▼
       Candidate Deduplication
              │
              ▼
       Top-K Evidence / Context
              │
              ▼
       Gemini 3.6 Flash
              │
              ▼
       Citation Validation
              │
              ▼
       Grounded Answer + Sources""",
                language="text",
            )

            diagnostics = {}

            for key in (
                "query_profile",
                "fusion_weights",
                "retrieval_config",
                "semantic_candidates",
                "bm25_candidates",
                "retrieved_candidates",
                "context_chunks",
            ):
                if key in result:
                    diagnostics[key] = result[key]

            if diagnostics:
                st.markdown("#### Returned diagnostics")
                st.json(diagnostics)

            st.caption(
                "Only diagnostics actually returned by the backend are shown. "
                "The UI does not reconstruct or invent ranked candidate lists."
            )

        # ====================================================================
        # RAW RESULT
        # ====================================================================

        with tab_raw:
            st.caption(
                "Developer-facing structured response returned by the pipeline."
            )
            st.json(result)


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.caption(
    "HybridRAG · Hybrid Retrieval + Retrieval-Augmented Generation "
    "· Evidence-first technical QA · UI v2.0"
)
