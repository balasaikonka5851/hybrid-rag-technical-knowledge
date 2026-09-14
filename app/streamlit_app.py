"""
HybridRAG - Technical Knowledge Retrieval & Question Answering System

Streamlit demonstration interface for:
    - Hybrid retrieval
    - BM25 keyword search
    - Semantic vector search
    - Reciprocal Rank Fusion
    - Grounded generation
    - Citation validation
    - Source inspection
    - Retrieval diagnostics
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORTS
# ============================================================

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.citation_validator import (
    validate_citations,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HybridRAG",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        border: 1px solid rgba(128,128,128,.22);
        background: linear-gradient(135deg, rgba(127,127,127,.12), rgba(127,127,127,.04));
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: clamp(2.2rem, 5vw, 3.8rem);
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.04em;
        margin: 0;
    }
    .hero-subtitle {
        margin-top: .7rem;
        font-size: 1.05rem;
        opacity: .78;
        max-width: 850px;
    }

    .badge {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        border: 1px solid rgba(128,128,128,.28);
        font-size: .78rem;
        margin: .25rem .3rem .1rem 0;
    }

    .answer-card {
        padding: 1.4rem 1.5rem;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,.24);
        background: rgba(127,127,127,.045);
        margin: .5rem 0 1rem 0;
    }

    .source-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,.20);
        background: rgba(127,127,127,.035);
        margin-bottom: .7rem;
    }

    .small-muted {
        font-size: .82rem;
        opacity: .68;
    }

    .trust-row {
        display: flex;
        gap: .6rem;
        flex-wrap: wrap;
        margin: .7rem 0;
    }

    .trust-pill {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 999px;
        padding: .35rem .7rem;
        font-size: .82rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        padding: .8rem;
        border-radius: 14px;
    }

    textarea {
        border-radius: 14px !important;
    }

    .architecture {
        font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        font-size: .88rem;
        line-height: 1.65;
        padding: 1.15rem;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,.22);
        overflow-x: auto;
    }

    .footer {
        text-align: center;
        opacity: .6;
        font-size: .8rem;
        padding: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🔎 HybridRAG</div>
        <div class="hero-subtitle">
            Intelligent technical knowledge retrieval with hybrid search,
            evidence-grounded generation, and citation validation.
        </div>
        <div class="trust-row">
            <span class="badge">🧠 Semantic Search</span>
            <span class="badge">🔤 BM25</span>
            <span class="badge">🔀 Weighted RRF</span>
            <span class="badge">📚 Evidence First</span>
            <span class="badge">🔗 Citation Validation</span>
            <span class="badge">🛡️ Grounded Answers</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ System")

    st.markdown("### Knowledge Base")
    st.success("● Online • validated at pipeline startup")
    st.write("**Corpus:** Python + FastAPI")
    st.write("**Chunks:** 18,150")
    st.write("**Embeddings:** 384-D MiniLM")
    st.write("**Index:** FAISS + BM25")

    st.markdown("### Retrieval")
    st.write("Semantic candidates: **60**")
    st.write("BM25 candidates: **60**")
    st.write("Weights: **0.75 / 0.25**")
    st.write("Fusion: **Weighted RRF (k=60)**")

    st.markdown("### Generation")
    st.write("Model: **Gemini 3.6 Flash**")
    st.write("Context window: **5 chunks**")

    st.markdown("### Guardrails")
    st.write("✓ Citation validation")
    st.write("✓ Evidence inspection")
    st.write("✓ Groundedness evaluation")
    st.write("✓ Pipeline error handling")

    st.divider()
    st.caption("Tip: ask for an API, concept, behavior, parameter, or example from the indexed documentation.")

# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource(show_spinner="Initializing HybridRAG…")
def get_pipeline():
    """
    Initialize the complete RAG stack once per Streamlit process.
    This prevents FAISS/BM25/model/knowledge-base initialization on
    every Streamlit rerun.
    """
    return RAGPipeline()


try:
    pipeline = get_pipeline()
    pipeline_ready = True
except Exception as error:
    pipeline_ready = False
    st.error("Unable to initialize the RAG pipeline.")
    st.exception(error)
    st.stop()


# PROJECT OVERVIEW
# ============================================================

with st.expander(
    "📘 About This Project",
    expanded=True,
):

    st.markdown(
        """
        ### HybridRAG

        HybridRAG combines **keyword-based retrieval** and
        **semantic vector retrieval** to find relevant
        technical documentation before passing the evidence
        to a generative language model.

        The objective is to improve:

        - Retrieval quality
        - Technical answer relevance
        - Evidence traceability
        - Hallucination resistance
        - Source transparency
        """
    )

    st.markdown(
        """
        ### Architecture
        """
    )

    st.markdown(
        """
        <div class="architecture">

        User Query
        <br>↓
        <br>┌───────────────────────────────┐
        <br>│ Semantic Retrieval            │
        <br>│ SentenceTransformer + FAISS  │
        <br>└───────────────────────────────┘
        <br>↓
        <br>┌───────────────────────────────┐
        <br>│ Keyword Retrieval             │
        <br>│ BM25                          │
        <br>└───────────────────────────────┘
        <br>↓
        <br>Weighted Reciprocal Rank Fusion
        <br>↓
        <br>Top Evidence
        <br>↓
        <br>Gemini
        <br>↓
        <br>Answer + Citations
        <br>↓
        <br>Citation / Groundedness Validation

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# KNOWLEDGE BASE DASHBOARD
# ============================================================

st.subheader("📊 Knowledge Base")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Logical Documents",
    "696",
)

col2.metric(
    "Chunks",
    "18,150",
)

col3.metric(
    "Embedding Dimension",
    "384",
)

col4.metric(
    "Technologies",
    "2",
)


kb1, kb2, kb3, kb4 = st.columns(4)

kb1.metric(
    "Python Documents",
    "536",
)

kb2.metric(
    "FastAPI Documents",
    "160",
)

kb3.metric(
    "Chunk Size",
    "1,200 chars",
)

kb4.metric(
    "Chunk Overlap",
    "200 chars",
)


# ============================================================
# RETRIEVAL BENCHMARK
# ============================================================

with st.expander(
    "📈 Retrieval Evaluation Results",
    expanded=False,
):

    st.markdown(
        """
        The retrieval benchmark was evaluated using
        **40 technical queries** with manually audited
        relevant sources.
        """
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
        """
        **Key observation:** Hybrid retrieval provides
        stronger deeper-rank recall than either individual
        retrieval method, reaching **87.5% Hit@10** in the
        candidate-depth experiment.
        """
    )

    st.caption(
        "MRR = Mean Reciprocal Rank"
    )


# ============================================================
# EXPERIMENTS
# ============================================================

with st.expander(
    "🧪 Retrieval Experiments",
    expanded=False,
):

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

    st.markdown(
    """
    A Cross-Encoder reranking stage was also evaluated.

    The MS-MARCO reranker improved Hit@1 slightly,
    but reduced deeper retrieval recall and overall MRR.
    Therefore it is retained as an experiment rather
    than being used as the primary production ranking stage.
    """
)

    st.warning(
        "Reranking was intentionally not selected as the primary ranking stage because it reduced MRR."
    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.subheader("💬 Ask the Technical Knowledge Base")

examples = [
    "How do I create dependencies in FastAPI?",
    "How does Python handle exceptions?",
    "How can I validate data received by an API?",
    "What is RequestValidationError in FastAPI?",
    "How does Python's yield keyword work?",
]

selected_example = st.selectbox(
    "Example questions",
    ["Custom question"] + examples,
)

if selected_example != "Custom question":

    default_query = selected_example

else:

    default_query = ""


query = st.text_area(
    "Enter your technical question",
    value=default_query,
    height=90,
    placeholder=(
        "Example: How do dependencies work in FastAPI?"
    ),
)


# ============================================================
# ASK BUTTON
# ============================================================

ask_clicked = st.button(
    "🚀 Ask HybridRAG",
    type="primary",
    use_container_width=True,
)

# Keep the last successful answer available across Streamlit reruns.
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ============================================================
# EXECUTION
# ============================================================

if ask_clicked:
    if not query.strip():
        st.warning("Please enter a technical question.")
        st.stop()

    query = query.strip()

    if len(query) < 3:
        st.warning("Please provide a more descriptive question.")
        st.stop()

    if len(query) > 2000:
        st.warning("Please keep the question under 2,000 characters.")
        st.stop()

    with st.spinner("🔎 Retrieving evidence → validating context → generating answer…"):
        try:
            start_time = time.perf_counter()
            result = pipeline.ask(query)
            elapsed = time.perf_counter() - start_time
            st.session_state.last_result = result
            st.session_state.last_query = query
        except Exception as error:
            st.error("The RAG pipeline encountered an error.")
            st.exception(error)
            st.stop()

# ============================================================
# RESULT
# ============================================================

result = st.session_state.last_result

if result:
    query_display = st.session_state.last_query

    st.divider()

    st.markdown(
        f'<div class="small-muted">QUESTION</div><h3>{query_display}</h3>',
        unsafe_allow_html=True,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    citation_result = validate_citations(answer, sources)

    # Trust summary
    st.markdown("### 🛡️ Answer quality")
    t1, t2, t3, t4 = st.columns(4)

    t1.metric(
        "Evidence chunks",
        result.get("context_chunks", len(sources))
    )
    t2.metric(
        "Valid citations",
        len(citation_result.get("valid_citations", []))
    )
    t3.metric(
        "Invalid citations",
        len(citation_result.get("invalid_citations", []))
    )
    t4.metric(
        "Retrieved candidates",
        len(result.get(
            "retrieved_results",
            result.get("hybrid_results", [])
        ))
    )

    if citation_result.get("all_citations_valid", False):
        st.success("✓ All generated citations map to returned evidence.")
    elif citation_result.get("has_citations", False):
        st.warning("Some citations could not be matched to returned evidence.")
    else:
        st.warning("No source citations were detected in the generated answer.")

    # Answer
    st.markdown("### 🧠 Answer")
    with st.container():
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        if answer:
            st.markdown(answer)
        else:
            st.warning("No answer was generated.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Performance
    st.markdown("### ⚡ Performance")
    retrieval_latency = result.get("retrieval_latency_ms", 0)
    generation_latency = result.get("generation_latency_ms", 0)
    total_latency = result.get("total_latency_ms", 0)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Retrieval", f"{retrieval_latency:.0f} ms")
    p2.metric("Generation", f"{generation_latency / 1000:.1f} s")
    p3.metric("Total", f"{total_latency / 1000:.1f} s")
    p4.metric(
        "Sources",
        len(sources)
    )

    # Evidence + diagnostics tabs
    tab_sources, tab_retrieval, tab_raw = st.tabs(
        ["📚 Evidence & Sources", "🔬 Retrieval Diagnostics", "🧩 Raw Result"]
    )

    with tab_sources:
        if not sources:
            st.info("No source metadata returned.")
        else:
            for index, source in enumerate(sources, start=1):
                technology = source.get("technology", "unknown")
                source_name = source.get("source", "unknown")
                section = source.get("section", "General")
                score = source.get(
                    "rrf_score",
                    source.get("score", 0)
                )
                page = source.get("page")
                sheet = source.get("sheet")
                slide = source.get("slide")

                location = []
                if page is not None:
                    location.append(f"Page {page}")
                if sheet:
                    location.append(f"Sheet: {sheet}")
                if slide is not None:
                    location.append(f"Slide {slide}")

                title = f"[SOURCE {index}] {source_name}"
                with st.expander(title, expanded=(index == 1)):
                    m1, m2, m3 = st.columns(3)
                    m1.write(f"**Technology**  \\n{technology}")
                    m2.write(f"**Section**  \\n{section}")
                    m3.write(f"**RRF score**  \\n{score:.6f}")

                    if location:
                        st.caption(" · ".join(location))

                    source_text = source.get("text", "")
                    if source_text:
                        st.markdown("**Retrieved evidence**")
                        st.code(source_text, language="text")

    with tab_retrieval:
        semantic_results = result.get("semantic_results", [])
        bm25_results = result.get("bm25_results", [])
        hybrid_results = result.get("hybrid_results", [])

        r1, r2, r3 = st.columns(3)
        r1.metric("Semantic", len(semantic_results))
        r2.metric("BM25", len(bm25_results))
        r3.metric("Hybrid", len(hybrid_results))

        st.markdown("#### Top semantic results")
        for item in semantic_results[:5]:
            st.write(
                f"**Rank {item.get('rank', '?')}** · "
                f"Score `{item.get('score', 0):.4f}` · "
                f"{item.get('metadata', {}).get('source', 'unknown')}"
            )

        st.markdown("#### Top BM25 results")
        for item in bm25_results[:5]:
            st.write(
                f"**Rank {item.get('rank', '?')}** · "
                f"Score `{item.get('score', 0):.4f}` · "
                f"{item.get('metadata', {}).get('source', 'unknown')}"
            )

        st.markdown("#### Top hybrid results")
        for item in hybrid_results[:10]:
            st.write(
                f"**Rank {item.get('rank', '?')}** · "
                f"RRF `{item.get('rrf_score', 0):.6f}` · "
                f"{item.get('metadata', {}).get('source', 'unknown')}"
            )

    with tab_raw:
        st.json(result)

# ============================================================
# FOOTER
# ============================================================

st.divider()
st.markdown(
    '<div class="footer">HybridRAG • Hybrid Retrieval + Retrieval-Augmented Generation • Evidence-first technical QA</div>',
    unsafe_allow_html=True,
)


# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    HybridRAG • Hybrid Retrieval + Retrieval-Augmented Generation
    • Built for technical documentation
    """
)
