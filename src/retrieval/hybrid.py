"""
Hybrid Retrieval using Weighted Reciprocal Rank Fusion (RRF).

Combines:
1. BM25 keyword retrieval
2. Semantic FAISS retrieval

The implementation supports configurable weights so we can
experimentally determine the best BM25/Semantic balance.
"""

from __future__ import annotations

from typing import Any


DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_TOP_K = 20
DEFAULT_BM25_TOP_K = 20

try:
    from src.retrieval.bm25 import BM25Retriever, load_chunks
    from src.retrieval.semantic import SemanticRetriever
except ModuleNotFoundError:
    from .bm25 import BM25Retriever, load_chunks
    from .semantic import SemanticRetriever


class HybridRetriever:
    """
    Hybrid BM25 + semantic retriever using weighted RRF.
    """

    def __init__(
        self,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:

        if semantic_top_k <= 0:
            raise ValueError("semantic_top_k must be > 0")

        if bm25_top_k <= 0:
            raise ValueError("bm25_top_k must be > 0")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")

        if semantic_weight < 0:
            raise ValueError("semantic_weight must be >= 0")

        if bm25_weight < 0:
            raise ValueError("bm25_weight must be >= 0")

        if semantic_weight == 0 and bm25_weight == 0:
            raise ValueError(
                "At least one retrieval weight must be greater than 0"
            )

        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

        print()
        print("=" * 80)
        print("INITIALIZING HYBRID RETRIEVER")
        print("=" * 80)

        print()
        print("[1/2] Initializing semantic retriever...")

        self.semantic = SemanticRetriever()

        print()
        print("[2/2] Initializing BM25 retriever...")

        chunks = load_chunks()
        self.bm25 = BM25Retriever(chunks)

        print()
        print("[OK] Hybrid retriever ready.")
        print(f"  Semantic candidates : {self.semantic_top_k}")
        print(f"  BM25 candidates     : {self.bm25_top_k}")
        print(f"  Semantic weight     : {self.semantic_weight:.2f}")
        print(f"  BM25 weight         : {self.bm25_weight:.2f}")
        print(f"  RRF k               : {self.rrf_k}")

    @staticmethod
    def _validate_results(
        results: list[dict[str, Any]],
        method_name: str,
    ) -> None:
        """
        Validate that every retrieval result contains a chunk ID.
        """

        for result in results:
            chunk_id = result.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    f"{method_name} result is missing chunk_id."
                )

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Weighted Reciprocal Rank Fusion.

        Score:

            semantic_weight / (rrf_k + semantic_rank)
          + bm25_weight / (rrf_k + bm25_rank)

        Results are merged by chunk_id.
        """

        self._validate_results(semantic_results, "Semantic")
        self._validate_results(bm25_results, "BM25")

        fused: dict[str, dict[str, Any]] = {}

        # ---------------------------------------------------------
        # Semantic results
        # ---------------------------------------------------------

        for rank, result in enumerate(semantic_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["semantic_rank"] = rank
            fused[chunk_id]["semantic_score"] = result.get("score")

            fused[chunk_id]["rrf_score"] += (
                self.semantic_weight
                / (self.rrf_k + rank)
            )

        # ---------------------------------------------------------
        # BM25 results
        # ---------------------------------------------------------

        for rank, result in enumerate(bm25_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "rrf_score": 0.0,
                    "semantic_rank": None,
                    "semantic_score": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                }

            fused[chunk_id]["bm25_rank"] = rank
            fused[chunk_id]["bm25_score"] = result.get("score")

            fused[chunk_id]["rrf_score"] += (
                self.bm25_weight
                / (self.rrf_k + rank)
            )

        # ---------------------------------------------------------
        # Sort
        # ---------------------------------------------------------

        ranked = sorted(
            fused.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        # Add final rank
        for rank, result in enumerate(ranked, start=1):
            result["rank"] = rank

        return ranked

    def search(
        self,
        query: str,
    ) -> dict[str, Any]:
        """
        Search using semantic retrieval, BM25, and weighted RRF.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty")

        semantic_results = self.semantic.search(
            query,
            top_k=self.semantic_top_k,
        )

        bm25_results = self.bm25.search(
            query,
            top_k=self.bm25_top_k,
        )

        hybrid_results = self.reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
        )

        return {
            "query": query,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
        }


def print_results(
    title: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
) -> None:
    """
    Pretty-print retrieval results.
    """

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for result in results[:top_k]:

        metadata = result.get("metadata", {})

        technology = metadata.get("technology", "unknown")
        source = metadata.get("source", "unknown")
        section = metadata.get("section", "")

        print()
        print(
            f"Rank {result.get('rank', '-')}"
            f" | chunk={result.get('chunk_id', '-')}"
        )

        print(
            f"Score={result.get('score', result.get('rrf_score', 0)):.4f}"
        )

        print(f"Technology={technology}")
        print(f"Source={source}")

        if section:
            print(f"Section={section}")

        print(f"Text={result.get('text', '')[:400]}...")


def main() -> None:

    print("=" * 80)
    print("HYBRID RAG - WEIGHTED RRF RETRIEVAL")
    print("=" * 80)

    retriever = HybridRetriever(
        semantic_top_k=20,
        bm25_top_k=20,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
    )

    while True:

        try:
            query = input(
                "\nEnter query "
                "(or type 'exit' to quit): "
            ).strip()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() == "exit":
            print("Exiting.")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        try:
            result = retriever.search(query)

            print_results(
                "SEMANTIC RESULTS",
                result["semantic_results"],
                top_k=5,
            )

            print_results(
                "BM25 RESULTS",
                result["bm25_results"],
                top_k=5,
            )

            print_results(
                "WEIGHTED HYBRID RESULTS",
                result["hybrid_results"],
                top_k=10,
            )

        except Exception as exc:
            print()
            print(f"[ERROR] Retrieval failed: {exc}")


if __name__ == "__main__":
    main()