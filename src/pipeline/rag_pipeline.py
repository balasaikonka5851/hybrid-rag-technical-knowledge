from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import GeminiGenerator
from src.evaluation.citation_validator import validate_citations


class RAGPipeline:
    """
    End-to-end Hybrid RAG pipeline.

    Query
      -> Hybrid Retrieval
      -> Context Selection
      -> Gemini Generation
      -> Citation Validation
      -> Structured Result
    """

    def __init__(
        self,
        semantic_top_k: int = 60,
        bm25_top_k: int = 60,
        semantic_weight: float = 0.75,
        bm25_weight: float = 0.25,
        rrf_k: int = 60,
        context_chunks: int = 5,
        model: str = "gemini-3.6-flash",
    ):
        if context_chunks <= 0:
            raise ValueError(
                "context_chunks must be greater than 0."
            )

        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than 0."
            )

        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k must be greater than 0."
            )

        if not 0 <= semantic_weight <= 1:
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if not 0 <= bm25_weight <= 1:
            raise ValueError(
                "bm25_weight must be between 0 and 1."
            )

        if semantic_weight + bm25_weight <= 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0."
            )

        print("\n" + "=" * 80)
        print("INITIALIZING RAG PIPELINE")
        print("=" * 80)

        print("\n[1/2] Initializing hybrid retriever...")

        self.retriever = HybridRetriever(
            semantic_top_k=semantic_top_k,
            bm25_top_k=bm25_top_k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
            rrf_k=rrf_k,
        )

        print("\n[2/2] Initializing Gemini generator...")

        self.generator = GeminiGenerator(
            model=model,
        )

        self.context_chunks = context_chunks
        self.model = model

        print("\n[OK] RAG pipeline ready.")

    def ask(
        self,
        query: str,
    ) -> dict[str, Any]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        total_start = time.perf_counter()

        print("\n" + "=" * 80)
        print("RAG QUERY")
        print("=" * 80)

        print(f"Query: {query}")

        # ---------------------------------------------------------
        # 1. Retrieval
        # ---------------------------------------------------------

        retrieval_start = time.perf_counter()

        retrieval = self.retriever.search(query)

        retrieval_latency = (
            time.perf_counter() - retrieval_start
        )

        hybrid_results = retrieval["hybrid_results"]

        if not hybrid_results:
            return {
                "query": query,
                "answer": (
                    "I couldn't find enough information "
                    "in the provided documentation."
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
                    retrieval_latency * 1000,
                    2,
                ),
                "generation_latency_ms": 0.0,
                "total_latency_ms": round(
                    (time.perf_counter() - total_start)
                    * 1000,
                    2,
                ),
            }

        # ---------------------------------------------------------
        # 2. Context selection
        # ---------------------------------------------------------

        selected_results = hybrid_results[
            : self.context_chunks
        ]

        print(
            f"\n[OK] Retrieved candidates: "
            f"{len(hybrid_results)}"
        )

        print(
            f"[OK] Context chunks selected: "
            f"{len(selected_results)}"
        )

        # ---------------------------------------------------------
        # 3. Generation
        # ---------------------------------------------------------

        generation_start = time.perf_counter()

        answer = self.generator.generate(
            query=query,
            results=selected_results,
            max_chunks=self.context_chunks,
        )

        generation_latency = (
            time.perf_counter() - generation_start
        )

        # ---------------------------------------------------------
        # 4. Source metadata
        # ---------------------------------------------------------

        sources = []

        for result in selected_results:

            metadata = result.get(
                "metadata",
                {},
            )

            # Hybrid RRF score.
            score = result.get(
                "rrf_score"
            )

            # Some implementations may store it
            # under "score", so keep that as fallback.
            if score is None:
                score = result.get(
                    "score"
                )

            sources.append(
    {
        "rank": result.get(
            "rank"
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
)

        # ---------------------------------------------------------
        # 5. Citation validation
        # ---------------------------------------------------------

        citation_validation = validate_citations(
            answer=answer,
            sources=sources,
        )

        print("\n[OK] Citation validation:")

        print(
            f"  Citations found: "
            f"{citation_validation['citations_found']}"
        )

        print(
            f"  Valid citations: "
            f"{citation_validation['valid_citations']}"
        )

        print(
            f"  Invalid citations: "
            f"{citation_validation['invalid_citations']}"
        )

        # ---------------------------------------------------------
        # 6. Latency
        # ---------------------------------------------------------

        total_latency = (
            time.perf_counter() - total_start
        )

        print(
            f"\n[OK] Retrieval latency: "
            f"{retrieval_latency * 1000:.2f} ms"
        )

        print(
            f"[OK] Generation latency: "
            f"{generation_latency * 1000:.2f} ms"
        )

        print(
            f"[OK] Total latency: "
            f"{total_latency * 1000:.2f} ms"
        )

        # ---------------------------------------------------------
        # 7. Structured result
        # ---------------------------------------------------------

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "citation_validation": citation_validation,
            "retrieved_candidates": len(
                hybrid_results
            ),
            "context_chunks": len(
                selected_results
            ),
            "model": self.model,
            "retrieval_latency_ms": round(
                retrieval_latency * 1000,
                2,
            ),
            "generation_latency_ms": round(
                generation_latency * 1000,
                2,
            ),
            "total_latency_ms": round(
                total_latency * 1000,
                2,
            ),
        }