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

    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .project-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 1rem;
    }

    .metric-card {
        padding: 0.8rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
        text-align: center;
    }

    .architecture {
        font-family: monospace;
        font-size: 0.95rem;
        line-height: 1.7;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.25);
    }

    .source-box {
        padding: 0.9rem;
        border-radius: 10px;
        border: 1px solid rgba(128,128,128,0.20);
        margin-bottom: 0.7rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔎 HybridRAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Intelligent Technical Knowledge Retrieval and
    Question Answering System
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    st.markdown("### Knowledge Base")

    st.info(
        """
        **Current corpus**

        • Python Documentation  
        • FastAPI Documentation

        The ingestion architecture also supports
        additional technical document formats.
        """
    )

    st.markdown("### Retrieval")

    st.write(
        "Semantic top-K: **60**"
    )

    st.write(
        "BM25 top-K: **60**"
    )

    st.write(
        "Semantic weight: **0.75**"
    )

    st.write(
        "BM25 weight: **0.25**"
    )

    st.write(
        "Fusion: **Weighted RRF**"
    )

    st.markdown("### Generation")

    st.write(
        "LLM: **Gemini 3.6 Flash**"
    )

    st.write(
        "Context chunks: **5**"
    )

    st.markdown("### Validation")

    st.write(
        "✓ Citation validation"
    )

    st.write(
        "✓ Evidence inspection"
    )

    st.write(
        "✓ Groundedness evaluation"
    )


# ============================================================
# PIPELINE
# ============================================================

@st.cache_resource
def get_pipeline():

    return RAGPipeline()


try:

    pipeline = get_pipeline()

except Exception as error:

    st.error(
        "Unable to initialize the RAG pipeline."
    )

    st.exception(error)

    st.stop()


# ============================================================
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


# ============================================================
# EXECUTION
# ============================================================

if ask_clicked:

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

        st.stop()

    query = query.strip()

    with st.spinner(
        "Searching documentation and generating grounded answer..."
    ):

        try:

            start_time = time.perf_counter()

            result = pipeline.ask(
                query
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

        except Exception as error:

            st.error(
                "The RAG pipeline encountered an error."
            )

            st.exception(error)

            st.stop()

    # ========================================================
    # ANSWER
    # ========================================================

    st.divider()

    st.subheader("🧠 Answer")

    answer = result.get(
        "answer",
        "",
    )

    if answer:

        st.markdown(
            answer
        )

    else:

        st.warning(
            "No answer was generated."
        )


    # ========================================================
    # PERFORMANCE
    # ========================================================

    st.subheader(
        "⚡ Performance"
    )

    retrieval_latency = result.get(
        "retrieval_latency_ms",
        0,
    )

    generation_latency = result.get(
        "generation_latency_ms",
        0,
    )

    total_latency = result.get(
        "total_latency_ms",
        elapsed * 1000,
    )

    p1, p2, p3, p4 = st.columns(4)

    p1.metric(
        "Retrieval",
        f"{retrieval_latency:.0f} ms",
    )

    p2.metric(
        "Generation",
        f"{generation_latency / 1000:.1f} s",
    )

    p3.metric(
        "Total",
        f"{total_latency / 1000:.1f} s",
    )

    p4.metric(
        "Candidates",
        len(
            result.get(
                "retrieved_results",
                result.get(
                    "hybrid_results",
                    [],
                ),
            )
        ),
    )


    # ========================================================
    # CITATION VALIDATION
    # ========================================================

    st.subheader(
        "🔗 Citation Validation"
    )

    sources = result.get(
        "sources",
        [],
    )

    citation_result = validate_citations(
        answer,
        sources,
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Citations Found",
        len(
            citation_result.get(
                "citations_found",
                [],
            )
        ),
    )

    c2.metric(
        "Valid Citations",
        len(
            citation_result.get(
                "valid_citations",
                [],
            )
        ),
    )

    c3.metric(
        "Invalid Citations",
        len(
            citation_result.get(
                "invalid_citations",
                [],
            )
        ),
    )

    if citation_result.get(
        "all_citations_valid",
        False,
    ):

        st.success(
            "✓ All generated source citations are valid."
        )

    elif citation_result.get(
        "has_citations",
        False,
    ):

        st.warning(
            "Some citations could not be matched to available sources."
        )

    else:

        st.warning(
            "No source citations were detected."
        )


    # ========================================================
    # SOURCES
    # ========================================================

    st.subheader(
        "📚 Evidence & Sources"
    )

    if not sources:

        st.info(
            "No source metadata returned."
        )

    else:

        for index, source in enumerate(
            sources,
            start=1,
        ):

            technology = source.get(
                "technology",
                "unknown",
            )

            source_name = source.get(
                "source",
                "unknown",
            )

            section = source.get(
                "section",
                "General",
            )

            score = source.get(
                "rrf_score",
                source.get(
                    "score",
                    0,
                ),
            )

            page = source.get(
                "page"
            )

            sheet = source.get(
                "sheet"
            )

            slide = source.get(
                "slide"
            )

            location = ""

            if page is not None:
                location += (
                    f" · Page {page}"
                )

            if sheet:
                location += (
                    f" · Sheet: {sheet}"
                )

            if slide is not None:
                location += (
                    f" · Slide {slide}"
                )

            with st.expander(
                f"[SOURCE {index}] {source_name}"
            ):

                m1, m2, m3 = st.columns(3)

                m1.write(
                    f"**Technology:** {technology}"
                )

                m2.write(
                    f"**Section:** {section}"
                )

                m3.write(
                    f"**RRF Score:** {score:.6f}"
                )

                if location:

                    st.write(
                        f"**Location:** {location}"
                    )

                source_text = source.get(
                    "text",
                    "",
                )

                if source_text:

                    st.markdown(
                        "**Retrieved evidence:**"
                    )

                    st.code(
                        source_text,
                        language="text",
                    )


    # ========================================================
    # RETRIEVAL INSPECTION
    # ========================================================

    with st.expander(
        "🔬 Retrieval Diagnostics",
        expanded=False,
    ):

        semantic_results = result.get(
            "semantic_results",
            [],
        )

        bm25_results = result.get(
            "bm25_results",
            [],
        )

        hybrid_results = result.get(
            "hybrid_results",
            [],
        )

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Semantic Results",
            len(semantic_results),
        )

        r2.metric(
            "BM25 Results",
            len(bm25_results),
        )

        r3.metric(
            "Hybrid Candidates",
            len(hybrid_results),
        )

        st.markdown(
            "### Top Semantic Results"
        )

        for item in semantic_results[:5]:

            st.write(
                f"**Rank {item.get('rank', '?')}** "
                f"| Score: "
                f"{item.get('score', 0):.4f} "
                f"| {item.get('metadata', {}).get('source', 'unknown')}"
            )

        st.markdown(
            "### Top BM25 Results"
        )

        for item in bm25_results[:5]:

            st.write(
                f"**Rank {item.get('rank', '?')}** "
                f"| Score: "
                f"{item.get('score', 0):.4f} "
                f"| {item.get('metadata', {}).get('source', 'unknown')}"
            )

        st.markdown(
            "### Top Hybrid Results"
        )

        for item in hybrid_results[:10]:

            st.write(
                f"**Rank {item.get('rank', '?')}** "
                f"| RRF: "
                f"{item.get('rrf_score', 0):.6f} "
                f"| {item.get('metadata', {}).get('source', 'unknown')}"
            )


    # ========================================================
    # RAW RESULT
    # ========================================================

    with st.expander(
        "🧩 Raw Pipeline Result",
        expanded=False,
    ):

        st.json(
            result
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    """
    HybridRAG • Hybrid Retrieval + Retrieval-Augmented Generation
    • Built for technical documentation
    """
)