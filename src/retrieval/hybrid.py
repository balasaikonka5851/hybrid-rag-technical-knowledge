"""
HybridRAG - Hybrid Retrieval

Combines:

    1. Semantic retrieval using FAISS
    2. Keyword retrieval using BM25

using Reciprocal Rank Fusion (RRF).

Pipeline:

    Query
      |
      +----> FAISS semantic search
      |
      +----> BM25 keyword search
      |
      v
    RRF fusion
      |
      v
    Hybrid ranked results
"""

from __future__ import annotations

import sys
from typing import Any

from bm25 import BM25Retriever, load_chunks
from semantic import SemanticRetriever


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5

# RRF constant.
#
# Standard choice:
#
#       k = 60
#
# Larger k reduces the influence of very high rankings.
RRF_K = 60


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:
    """
    Hybrid retrieval using FAISS + BM25 + RRF.
    """

    def __init__(
        self,
        semantic_top_k: int = 20,
        bm25_top_k: int = 20,
    ) -> None:

        if semantic_top_k <= 0:
            raise ValueError(
                "semantic_top_k must be greater than zero."
            )

        if bm25_top_k <= 0:
            raise ValueError(
                "bm25_top_k must be greater than zero."
            )

        print()
        print("=" * 80)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 80)

        # ----------------------------------------------------
        # Semantic retriever
        # ----------------------------------------------------

        print()
        print("[1/3] Initializing semantic retriever...")

        self.semantic_retriever = (
            SemanticRetriever()
        )

        # ----------------------------------------------------
        # BM25 retriever
        # ----------------------------------------------------

        print()
        print("[2/3] Initializing BM25 retriever...")

        chunks = load_chunks()

        self.bm25_retriever = BM25Retriever(
            chunks
        )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        print()
        print("[3/3] Configuring RRF...")

        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = RRF_K

        print(
            f"[OK] Semantic candidates: "
            f"{semantic_top_k}"
        )

        print(
            f"[OK] BM25 candidates: "
            f"{bm25_top_k}"
        )

        print(
            f"[OK] RRF k: "
            f"{self.rrf_k}"
        )

        print()
        print(
            "[OK] Hybrid retriever ready."
        )

    # ========================================================
    # RRF
    # ========================================================

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Combine semantic and BM25 rankings using RRF.

        Formula:

            RRF(d) =
                1 / (k + rank_semantic)
                +
                1 / (k + rank_bm25)

        A document appearing in both rankings receives
        contributions from both systems.
        """

        # ----------------------------------------------------
        # Store fused information by chunk ID.
        # ----------------------------------------------------

        fused: dict[str, dict[str, Any]] = {}

        # ----------------------------------------------------
        # Semantic contribution
        # ----------------------------------------------------

        for result in semantic_results:

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_rank": None,
                    "semantic_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            rank = result["rank"]

            fused[chunk_id][
                "semantic_rank"
            ] = rank

            fused[chunk_id][
                "semantic_score"
            ] = result["score"]

            fused[chunk_id][
                "rrf_score"
            ] += 1.0 / (
                self.rrf_k + rank
            )

        # ----------------------------------------------------
        # BM25 contribution
        # ----------------------------------------------------

        for result in bm25_results:

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:

                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "semantic_rank": None,
                    "semantic_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0,
                }

            rank = result["rank"]

            fused[chunk_id][
                "bm25_rank"
            ] = rank

            fused[chunk_id][
                "bm25_score"
            ] = result["score"]

            fused[chunk_id][
                "rrf_score"
            ] += 1.0 / (
                self.rrf_k + rank
            )

        # ----------------------------------------------------
        # Sort by RRF score
        # ----------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # ----------------------------------------------------
        # Add final rank
        # ----------------------------------------------------

        for rank, result in enumerate(
            ranked,
            start=1,
        ):

            result["rank"] = rank

        return ranked

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """
        Run semantic retrieval, BM25 retrieval,
        and RRF fusion.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # Semantic search
        # ----------------------------------------------------

        semantic_results = (
            self.semantic_retriever.search(
                query=query,
                top_k=self.semantic_top_k,
            )
        )

        # ----------------------------------------------------
        # BM25 search
        # ----------------------------------------------------

        bm25_results = (
            self.bm25_retriever.search(
                query=query,
                top_k=self.bm25_top_k,
            )
        )

        # ----------------------------------------------------
        # RRF
        # ----------------------------------------------------

        hybrid_results = (
            self.reciprocal_rank_fusion(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
            )
        )

        # ----------------------------------------------------
        # Final top-k
        # ----------------------------------------------------

        hybrid_results = hybrid_results[
            : min(
                top_k,
                len(hybrid_results),
            )
        ]

        return {
            "query": query,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
        }


# ============================================================
# DISPLAY
# ============================================================

def display_hybrid_results(
    output: dict[str, Any],
) -> None:

    query = output["query"]

    semantic_results = output[
        "semantic_results"
    ]

    bm25_results = output[
        "bm25_results"
    ]

    hybrid_results = output[
        "hybrid_results"
    ]

    # --------------------------------------------------------
    # Semantic ranking
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("SEMANTIC TOP RESULTS")
    print("=" * 90)

    for result in semantic_results[:5]:

        metadata = result["metadata"]

        print(
            f"{result['rank']:>2}. "
            f"score={result['score']:.4f} | "
            f"{metadata['technology']:<7} | "
            f"{metadata['source']}"
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("BM25 TOP RESULTS")
    print("=" * 90)

    for result in bm25_results[:5]:

        metadata = result["metadata"]

        print(
            f"{result['rank']:>2}. "
            f"score={result['score']:.4f} | "
            f"{metadata['technology']:<7} | "
            f"{metadata['source']}"
        )

    # --------------------------------------------------------
    # Hybrid ranking
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("HYBRID RRF TOP RESULTS")
    print("=" * 90)

    for result in hybrid_results:

        metadata = result["metadata"]

        semantic_rank = (
            result["semantic_rank"]
            if result["semantic_rank"] is not None
            else "-"
        )

        bm25_rank = (
            result["bm25_rank"]
            if result["bm25_rank"] is not None
            else "-"
        )

        print()
        print(
            f"Rank {result['rank']}"
        )

        print(
            f"RRF score      : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Semantic rank  : "
            f"{semantic_rank}"
        )

        print(
            f"BM25 rank      : "
            f"{bm25_rank}"
        )

        print(
            f"Semantic score : "
            f"{result['semantic_score']}"
        )

        print(
            f"BM25 score     : "
            f"{result['bm25_score']}"
        )

        print(
            f"Technology     : "
            f"{metadata['technology']}"
        )

        print(
            f"Source         : "
            f"{metadata['source']}"
        )

        print(
            f"Section        : "
            f"{metadata['section']}"
        )

        print(
            f"Chunk ID       : "
            f"{result['chunk_id']}"
        )

        print()

        text = result["text"]

        if len(text) > 600:
            text = text[:600] + "..."

        print(text)

    print()
    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 90)
    print("HYBRID RAG - HYBRID RETRIEVAL")
    print("=" * 90)

    try:

        retriever = HybridRetriever(
            semantic_top_k=20,
            bm25_top_k=20,
        )

        print()
        print("Hybrid retriever ready.")

        print()
        print(
            "Test queries:"
        )

        print(
            "  1. How do I create dependencies in FastAPI?"
        )

        print(
            "  2. RequestValidationError"
        )

        print(
            "  3. How can I validate incoming API data?"
        )

        print(
            "  4. Python exception handling"
        )

        print()
        print("Type 'exit' to stop.")

        while True:

            try:

                query = input(
                    "\nQuery: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print()
                print("Exiting.")

                break

            if query.lower() in {
                "exit",
                "quit",
            }:

                print("Exiting.")
                break

            if not query:

                print(
                    "[WARNING] "
                    "Please enter a question."
                )

                continue

            try:

                output = retriever.search(
                    query=query,
                    top_k=5,
                )

                display_hybrid_results(
                    output
                )

            except Exception as exc:

                print()
                print(
                    f"[ERROR] Search failed: {exc}"
                )

    except Exception as exc:

        print()
        print("=" * 90)
        print("HYBRID RETRIEVER FAILED")
        print("=" * 90)

        print()
        print(f"[ERROR] {exc}")

        sys.exit(1)


if __name__ == "__main__":
    main()