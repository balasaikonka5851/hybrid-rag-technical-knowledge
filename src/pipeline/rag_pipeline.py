"""
HybridRAG - Robust End-to-End RAG Pipeline

Pipeline:

    Query
      ↓
    Knowledge-Base Bootstrap + Validation
      ↓
    Hybrid Retrieval
      ├── Semantic Search
      └── BM25 Search
      ↓
    Reciprocal Rank Fusion
      ↓
    Candidate Deduplication
      ↓
    Context Selection
      ↓
    Gemini Generation
      ↓
    Citation Validation
      ↓
    Structured Result

Design goals:
    - Robust failure handling
    - Deterministic behavior
    - Strong input validation
    - Evidence preservation
    - Citation validation
    - Useful latency metrics
    - Model-independent retrieval layer
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# INTERNAL IMPORTS
# ============================================================

from src.deployment.artifact_loader import (
    ensure_knowledge_base,
    validate_knowledge_base,
)

from src.retrieval.hybrid import (
    HybridRetriever,
)

from src.generation.generator import (
    GeminiGenerator,
)

from src.evaluation.citation_validator import (
    validate_citations,
)


class RAGPipeline:
    """
    Robust end-to-end Hybrid RAG pipeline.

    The pipeline intentionally separates:

        Retrieval
        Generation
        Validation

    so that the LLM can later be replaced without
    rebuilding the retrieval system.
    """

    # ========================================================
    # DEFAULT CONFIGURATION
    # ========================================================

    DEFAULT_SEMANTIC_TOP_K = 60
    DEFAULT_BM25_TOP_K = 60

    DEFAULT_SEMANTIC_WEIGHT = 0.75
    DEFAULT_BM25_WEIGHT = 0.25

    DEFAULT_RRF_K = 60

    DEFAULT_CONTEXT_CHUNKS = 5

    DEFAULT_MODEL = "gemini-3.6-flash"

    MIN_QUERY_LENGTH = 2
    MAX_QUERY_LENGTH = 2000

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
        context_chunks: int = DEFAULT_CONTEXT_CHUNKS,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize the complete RAG pipeline.

        Raises:
            ValueError:
                If configuration parameters are invalid.

            FileNotFoundError / ValueError:
                If the knowledge base is missing or corrupted.
        """

        # ----------------------------------------------------
        # Validate configuration
        # ----------------------------------------------------

        self._validate_configuration(
            semantic_top_k=semantic_top_k,
            bm25_top_k=bm25_top_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
            context_chunks=context_chunks,
            model=model,
        )

        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k

        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        self.rrf_k = rrf_k
        self.context_chunks = context_chunks

        self.model = model

        # ----------------------------------------------------
        # Pipeline initialization
        # ----------------------------------------------------

        print("\n" + "=" * 80)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 80)

        # ----------------------------------------------------
        # 0. Knowledge-base bootstrap + validation
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # A deployment environment starts without generated
        # RAG files because data/processed/ and vectorstore/
        # are intentionally excluded from Git.
        #
        # ensure_knowledge_base() handles both cases:
        #   1. KB exists locally -> validate/reuse it.
        #   2. KB is missing -> download the verified release
        #      artifact, extract it safely, then validate it.
        #
        # validate_knowledge_base() alone is NOT sufficient for
        # Streamlit Cloud because a fresh machine has no KB yet.
        # ----------------------------------------------------

        print("\n[0/2] Preparing knowledge base...")

        validation_start = time.perf_counter()

        try:
            # Bootstrap first. This is safe to call even when the
            # knowledge base already exists locally.
            ensure_knowledge_base()

            # Validate the actual files after bootstrap/reuse.
            self.knowledge_base_info = validate_knowledge_base()

        except Exception as error:
            print(
                "\n[ERROR] Knowledge-base initialization failed:"
            )
            print(
                f"  {type(error).__name__}: {error}"
            )
            raise

        validation_latency = (
            time.perf_counter() - validation_start
        )

        self.knowledge_base_validation_latency_ms = round(
            validation_latency * 1000,
            2,
        )

        print(
            f"[OK] Knowledge base ready: "
            f"{self.knowledge_base_info['chunks']:,} chunks"
        )

        print(
            f"[OK] Knowledge-base bootstrap/validation latency: "
            f"{self.knowledge_base_validation_latency_ms:.2f} ms"
        )

        # ----------------------------------------------------
        # 1. Hybrid retriever
        # ----------------------------------------------------

        print(
            "\n[1/2] Initializing hybrid retriever..."
        )

        retriever_start = time.perf_counter()

        self.retriever = HybridRetriever(
            semantic_top_k=self.semantic_top_k,
            bm25_top_k=self.bm25_top_k,
            semantic_weight=self.semantic_weight,
            bm25_weight=self.bm25_weight,
            rrf_k=self.rrf_k,
        )

        retriever_latency = (
            time.perf_counter() - retriever_start
        )

        self.retriever_initialization_latency_ms = round(
            retriever_latency * 1000,
            2,
        )

        # ----------------------------------------------------
        # 2. Generator
        # ----------------------------------------------------

        print(
            "\n[2/2] Initializing Gemini generator..."
        )

        generator_start = time.perf_counter()

        self.generator = GeminiGenerator(
            model=self.model,
        )

        generator_latency = (
            time.perf_counter() - generator_start
        )

        self.generator_initialization_latency_ms = round(
            generator_latency * 1000,
            2,
        )

        print("\n[OK] RAG pipeline ready.")

        print(
            f"  Knowledge-base chunks : "
            f"{self.knowledge_base_info['chunks']:,}"
        )

        print(
            f"  Semantic candidates   : "
            f"{self.semantic_top_k}"
        )

        print(
            f"  BM25 candidates       : "
            f"{self.bm25_top_k}"
        )

        print(
            f"  Semantic weight       : "
            f"{self.semantic_weight}"
        )

        print(
            f"  BM25 weight           : "
            f"{self.bm25_weight}"
        )

        print(
            f"  Context chunks        : "
            f"{self.context_chunks}"
        )

        print(
            f"  Model                 : "
            f"{self.model}"
        )

    # ========================================================
    # CONFIGURATION VALIDATION
    # ========================================================

    @staticmethod
    def _validate_configuration(
        semantic_top_k: int,
        bm25_top_k: int,
        semantic_weight: float,
        bm25_weight: float,
        rrf_k: int,
        context_chunks: int,
        model: str,
    ) -> None:
        """Validate all pipeline configuration parameters."""

        if not isinstance(
            semantic_top_k,
            int,
        ):
            raise TypeError(
                "semantic_top_k must be an integer."
            )

        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than 0."
            )

        if not isinstance(
            bm25_top_k,
            int,
        ):
            raise TypeError(
                "bm25_top_k must be an integer."
            )

        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k must be greater than 0."
            )

        if not isinstance(
            context_chunks,
            int,
        ):
            raise TypeError(
                "context_chunks must be an integer."
            )

        if context_chunks <= 0:
            raise ValueError(
                "context_chunks must be greater than 0."
            )

        if context_chunks > (
            semantic_top_k + bm25_top_k
        ):
            raise ValueError(
                "context_chunks cannot exceed the "
                "combined retrieval candidate limit."
            )

        if not isinstance(
            rrf_k,
            int,
        ):
            raise TypeError(
                "rrf_k must be an integer."
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0."
            )

        if not isinstance(
            semantic_weight,
            (int, float),
        ):
            raise TypeError(
                "semantic_weight must be numeric."
            )

        if not 0 <= semantic_weight <= 1:
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if not isinstance(
            bm25_weight,
            (int, float),
        ):
            raise TypeError(
                "bm25_weight must be numeric."
            )

        if not 0 <= bm25_weight <= 1:
            raise ValueError(
                "bm25_weight must be between 0 and 1."
            )

        if (
            semantic_weight + bm25_weight
            <= 0
        ):
            raise ValueError(
                "At least one retrieval weight "
                "must be greater than 0."
            )

        if not isinstance(
            model,
            str,
        ):
            raise TypeError(
                "model must be a string."
            )

        model = model.strip()

        if not model:
            raise ValueError(
                "model cannot be empty."
            )

    # ========================================================
    # QUERY VALIDATION
    # ========================================================

    @classmethod
    def _validate_query(
        cls,
        query: str,
    ) -> str:
        """
        Validate and normalize a user query.
        """

        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        query = " ".join(
            query.strip().split()
        )

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        if len(query) < cls.MIN_QUERY_LENGTH:
            raise ValueError(
                f"query must contain at least "
                f"{cls.MIN_QUERY_LENGTH} characters."
            )

        if len(query) > cls.MAX_QUERY_LENGTH:
            raise ValueError(
                f"query exceeds the maximum allowed "
                f"length of {cls.MAX_QUERY_LENGTH} characters."
            )

        return query

    # ========================================================
    # RESULT DEDUPLICATION
    # ========================================================

    @staticmethod
    def _deduplicate_results(
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate chunks while preserving ranking order.

        chunk_id is preferred as the unique identifier.
        Text is used as a fallback.
        """

        unique_results = []

        seen_ids: set[str] = set()
        seen_text: set[str] = set()

        for result in results:

            if not isinstance(
                result,
                dict,
            ):
                continue

            chunk_id = result.get(
                "chunk_id"
            )

            text = result.get(
                "text",
                "",
            )

            if chunk_id is not None:

                key = str(
                    chunk_id
                )

                if key in seen_ids:
                    continue

                seen_ids.add(key)

            else:

                normalized_text = (
                    str(text)
                    .strip()
                    .lower()
                )

                if (
                    normalized_text
                    and normalized_text
                    in seen_text
                ):
                    continue

                if normalized_text:
                    seen_text.add(
                        normalized_text
                    )

            unique_results.append(
                result
            )

        return unique_results

    # ========================================================
    # CONTEXT SELECTION
    # ========================================================

    def _select_context(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Select the highest-ranked unique results.

        This preserves the existing retrieval ranking while
        preventing duplicate chunks from consuming context.
        """

        unique_results = (
            self._deduplicate_results(
                results
            )
        )

        selected = unique_results[
            : self.context_chunks
        ]

        return selected

    # ========================================================
    # SAFE ERROR RESULT
    # ========================================================

    def _build_error_result(
        self,
        query: str,
        error: Exception,
        retrieval_latency_ms: float = 0.0,
        generation_latency_ms: float = 0.0,
        total_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """
        Build a consistent structured error response.
        """

        return {
            "query": query,
            "answer": (
                "I was unable to complete the request "
                "because the RAG pipeline encountered "
                "an internal error. Please try again."
            ),
            "sources": [],
            "citation_validation": {
                "citations_found": [],
                "valid_citations": [],
                "invalid_citations": [],
                "citation_count": 0,
                "valid_citation_count": 0,
                "invalid_citation_count": 0,
                "has_citations": False,
                "all_citations_valid": True,
                "cited_sources": [],
            },
            "retrieved_candidates": 0,
            "context_chunks": 0,
            "model": self.model,
            "retrieval_latency_ms": round(
                retrieval_latency_ms,
                2,
            ),
            "generation_latency_ms": round(
                generation_latency_ms,
                2,
            ),
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),
            "success": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

    # ========================================================
    # MAIN QUERY METHOD
    # ========================================================

    def ask(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Execute one complete RAG query.

        Returns a structured dictionary containing:

            answer
            sources
            citation validation
            retrieval statistics
            timing statistics
            model information
            success/error information
        """

        # ----------------------------------------------------
        # Query validation
        # ----------------------------------------------------

        query = self._validate_query(
            query
        )

        total_start = time.perf_counter()

        print("\n" + "=" * 80)
        print("RAG QUERY")
        print("=" * 80)

        print(
            f"Query: {query}"
        )

        # ====================================================
        # 1. RETRIEVAL
        # ====================================================

        retrieval_start = time.perf_counter()

        try:

            retrieval = self.retriever.search(
                query
            )

        except Exception as error:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            print(
                "\n[ERROR] Retrieval failed:"
            )

            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    time.perf_counter()
                    - retrieval_start
                )
                * 1000,
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        retrieval_latency = (
            time.perf_counter()
            - retrieval_start
        )

        # ----------------------------------------------------
        # Validate retrieval response
        # ----------------------------------------------------

        if not isinstance(
            retrieval,
            dict,
        ):
            error = TypeError(
                "Retriever returned an invalid "
                "response type."
            )

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    retrieval_latency
                    * 1000
                ),
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        hybrid_results = retrieval.get(
            "hybrid_results",
            [],
        )

        if hybrid_results is None:
            hybrid_results = []

        if not isinstance(
            hybrid_results,
            list,
        ):
            error = TypeError(
                "Retriever returned invalid "
                "hybrid_results."
            )

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    retrieval_latency
                    * 1000
                ),
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        # ----------------------------------------------------
        # No retrieval result
        # ----------------------------------------------------

        if not hybrid_results:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            print(
                "\n[WARN] No relevant retrieval "
                "candidates found."
            )

            return {
                "query": query,
                "answer": (
                    "I couldn't find enough information "
                    "in the provided knowledge base to "
                    "answer this question reliably."
                ),
                "sources": [],
                "citation_validation": {
                    "citations_found": [],
                    "valid_citations": [],
                    "invalid_citations": [],
                    "citation_count": 0,
                    "valid_citation_count": 0,
                    "invalid_citation_count": 0,
                    "has_citations": False,
                    "all_citations_valid": True,
                    "cited_sources": [],
                },
                "retrieved_candidates": 0,
                "context_chunks": 0,
                "model": self.model,
                "retrieval_latency_ms": round(
                    retrieval_latency
                    * 1000,
                    2,
                ),
                "generation_latency_ms": 0.0,
                "total_latency_ms": round(
                    total_latency
                    * 1000,
                    2,
                ),
                "success": True,
                "answerable": False,
                "error_type": None,
                "error_message": None,
            }

        print(
            f"\n[OK] Retrieved candidates: "
            f"{len(hybrid_results)}"
        )

        # ====================================================
        # 2. CONTEXT SELECTION
        # ====================================================

        selected_results = (
            self._select_context(
                hybrid_results
            )
        )

        if not selected_results:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            error = ValueError(
                "Retrieval returned candidates "
                "but no valid context chunks."
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    retrieval_latency
                    * 1000
                ),
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        print(
            f"[OK] Context chunks selected: "
            f"{len(selected_results)}"
        )

        # ====================================================
        # 3. GENERATION
        # ====================================================

        generation_start = (
            time.perf_counter()
        )

        try:

            answer = self.generator.generate(
                query=query,
                results=selected_results,
                max_chunks=self.context_chunks,
            )

        except Exception as error:

            generation_latency = (
                time.perf_counter()
                - generation_start
            )

            total_latency = (
                time.perf_counter()
                - total_start
            )

            print(
                "\n[ERROR] Generation failed:"
            )

            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    retrieval_latency
                    * 1000
                ),
                generation_latency_ms=(
                    generation_latency
                    * 1000
                ),
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        generation_latency = (
            time.perf_counter()
            - generation_start
        )

        if not isinstance(
            answer,
            str,
        ):
            answer = str(
                answer
            )

        answer = answer.strip()

        if not answer:

            error = ValueError(
                "Generator returned an empty answer."
            )

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return self._build_error_result(
                query=query,
                error=error,
                retrieval_latency_ms=(
                    retrieval_latency
                    * 1000
                ),
                generation_latency_ms=(
                    generation_latency
                    * 1000
                ),
                total_latency_ms=(
                    total_latency
                    * 1000
                ),
            )

        # ====================================================
        # 4. SOURCE METADATA
        # ====================================================

        sources: list[
            dict[str, Any]
        ] = []

        for position, result in enumerate(
            selected_results,
            start=1,
        ):

            metadata = result.get(
                "metadata",
                {},
            )

            if not isinstance(
                metadata,
                dict,
            ):
                metadata = {}

            score = result.get(
                "rrf_score"
            )

            if score is None:
                score = result.get(
                    "score"
                )

            source = {
                "rank": result.get(
                    "rank",
                    position,
                ),
                "chunk_id": result.get(
                    "chunk_id"
                ),
                "technology": metadata.get(
                    "technology",
                    "unknown",
                ),
                "source": metadata.get(
                    "source",
                    "unknown",
                ),
                "section": metadata.get(
                    "section",
                    "",
                ),
                "score": score,
                "text": result.get(
                    "text",
                    "",
                ),
            }

            sources.append(
                source
            )

        # ====================================================
        # 5. CITATION VALIDATION
        # ====================================================

        try:

            citation_validation = (
                validate_citations(
                    answer=answer,
                    sources=sources,
                )
            )

        except Exception as error:

            print(
                "\n[WARN] Citation validation "
                "failed:"
            )

            print(
                f"  {type(error).__name__}: "
                f"{error}"
            )

            citation_validation = {
                "citations_found": [],
                "valid_citations": [],
                "invalid_citations": [],
                "citation_count": 0,
                "valid_citation_count": 0,
                "invalid_citation_count": 0,
                "has_citations": False,
                "all_citations_valid": False,
                "cited_sources": [],
                "validation_error": str(
                    error
                ),
            }

        print(
            "\n[OK] Citation validation:"
        )

        print(
            f"  Citations found: "
            f"{citation_validation.get('citations_found', [])}"
        )

        print(
            f"  Valid citations: "
            f"{citation_validation.get('valid_citations', [])}"
        )

        print(
            f"  Invalid citations: "
            f"{citation_validation.get('invalid_citations', [])}"
        )

        # ====================================================
        # 6. LATENCY
        # ====================================================

        total_latency = (
            time.perf_counter()
            - total_start
        )

        retrieval_latency_ms = (
            retrieval_latency
            * 1000
        )

        generation_latency_ms = (
            generation_latency
            * 1000
        )

        total_latency_ms = (
            total_latency
            * 1000
        )

        print(
            f"\n[OK] Retrieval latency: "
            f"{retrieval_latency_ms:.2f} ms"
        )

        print(
            f"[OK] Generation latency: "
            f"{generation_latency_ms:.2f} ms"
        )

        print(
            f"[OK] Total latency: "
            f"{total_latency_ms:.2f} ms"
        )

        # ====================================================
        # 7. RETRIEVAL STATISTICS
        # ====================================================

        semantic_results = retrieval.get(
            "semantic_results",
            [],
        )

        bm25_results = retrieval.get(
            "bm25_results",
            [],
        )

        # ====================================================
        # 8. FINAL STRUCTURED RESULT
        # ====================================================

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "citation_validation": citation_validation,

            # Retrieval
            "retrieved_candidates": len(
                hybrid_results
            ),
            "semantic_candidates": len(
                semantic_results
            )
            if isinstance(
                semantic_results,
                list,
            )
            else 0,
            "bm25_candidates": len(
                bm25_results
            )
            if isinstance(
                bm25_results,
                list,
            )
            else 0,

            # Context
            "context_chunks": len(
                selected_results
            ),

            # Model
            "model": self.model,

            # Knowledge base
            "knowledge_base_chunks": (
                self.knowledge_base_info[
                    "chunks"
                ]
            ),
            "embedding_dimension": (
                self.knowledge_base_info[
                    "dimension"
                ]
            ),

            # Retrieval configuration
            "semantic_weight": (
                self.semantic_weight
            ),
            "bm25_weight": (
                self.bm25_weight
            ),
            "rrf_k": self.rrf_k,

            # Timing
            "knowledge_base_validation_latency_ms": (
                self.knowledge_base_validation_latency_ms
            ),
            "retrieval_latency_ms": round(
                retrieval_latency_ms,
                2,
            ),
            "generation_latency_ms": round(
                generation_latency_ms,
                2,
            ),
            "total_latency_ms": round(
                total_latency_ms,
                2,
            ),

            # Status
            "success": True,
            "answerable": True,
            "error_type": None,
            "error_message": None,
        }


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    pipeline = RAGPipeline()

    result = pipeline.ask(
        "How do I create dependencies in FastAPI?"
    )

    print("\n" + "=" * 80)
    print("FINAL STRUCTURED RESULT")
    print("=" * 80)

    print(
        f"\nQUERY:\n{result['query']}"
    )

    print(
        f"\nANSWER:\n{result['answer']}"
    )

    print(
        "\nCITATION VALIDATION:"
    )

    print(
        result["citation_validation"]
    )

    print(
        "\nSTATISTICS:"
    )

    print(
        f"  Retrieved candidates: "
        f"{result['retrieved_candidates']}"
    )

    print(
        f"  Context chunks: "
        f"{result['context_chunks']}"
    )

    print(
        f"  Model: "
        f"{result['model']}"
    )

    print(
        f"  Retrieval latency: "
        f"{result['retrieval_latency_ms']} ms"
    )

    print(
        f"  Generation latency: "
        f"{result['generation_latency_ms']} ms"
    )

    print(
        f"  Total latency: "
        f"{result['total_latency_ms']} ms"
    )

    print(
        "\n[PASS] RAG pipeline completed."
    )