"""
Hybrid Retrieval using Query-Aware Weighted Reciprocal Rank Fusion.

Combines:
1. BM25 keyword retrieval
2. Semantic FAISS retrieval
3. Query-aware fusion weights

The retriever keeps the existing 60/60 candidate architecture and
uses adaptive weights only at fusion time. FAISS, the embedding model,
and BM25 are still initialized once and reused.
"""

from __future__ import annotations

from typing import Any

try:
    from src.retrieval.adaptive import QueryProfile, profile_query
    from src.retrieval.bm25 import BM25Retriever, load_chunks
    from src.retrieval.semantic import SemanticRetriever
except ModuleNotFoundError:
    from .adaptive import QueryProfile, profile_query
    from .bm25 import BM25Retriever, load_chunks
    from .semantic import SemanticRetriever


DEFAULT_RRF_K = 60
DEFAULT_SEMANTIC_TOP_K = 60
DEFAULT_BM25_TOP_K = 60
DEFAULT_SEMANTIC_WEIGHT = 0.75
DEFAULT_BM25_WEIGHT = 0.25


class HybridRetriever:
    """Hybrid BM25 + semantic retrieval using query-aware weighted RRF."""

    def __init__(
        self,
        semantic_top_k: int = DEFAULT_SEMANTIC_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        bm25_weight: float = DEFAULT_BM25_WEIGHT,
        rrf_k: int = DEFAULT_RRF_K,
        adaptive: bool = True,
    ) -> None:

        if semantic_top_k <= 0:
            raise ValueError("semantic_top_k must be > 0")
        if bm25_top_k <= 0:
            raise ValueError("bm25_top_k must be > 0")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be > 0")
        if semantic_weight < 0 or bm25_weight < 0:
            raise ValueError("retrieval weights must be >= 0")
        if semantic_weight == 0 and bm25_weight == 0:
            raise ValueError("At least one retrieval weight must be > 0")

        self.semantic_top_k = semantic_top_k
        self.bm25_top_k = bm25_top_k
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.adaptive = bool(adaptive)

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
        print(f"  Base semantic weight: {self.semantic_weight:.2f}")
        print(f"  Base BM25 weight    : {self.bm25_weight:.2f}")
        print(f"  RRF k               : {self.rrf_k}")
        print(f"  Adaptive fusion     : {'ON' if self.adaptive else 'OFF'}")

    @staticmethod
    def _validate_results(
        results: list[dict[str, Any]],
        method_name: str,
    ) -> None:
        for result in results:
            if not result.get("chunk_id"):
                raise ValueError(
                    f"{method_name} result is missing chunk_id."
                )

    def reciprocal_rank_fusion(
        self,
        semantic_results: list[dict[str, Any]],
        bm25_results: list[dict[str, Any]],
        *,
        semantic_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fuse two ranked lists using weighted reciprocal rank fusion."""

        self._validate_results(semantic_results, "Semantic")
        self._validate_results(bm25_results, "BM25")

        sw = self.semantic_weight if semantic_weight is None else semantic_weight
        bw = self.bm25_weight if bm25_weight is None else bm25_weight

        if sw < 0 or bw < 0 or (sw == 0 and bw == 0):
            raise ValueError("Invalid fusion weights.")

        total = sw + bw
        sw /= total
        bw /= total

        fused: dict[str, dict[str, Any]] = {}

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

            item = fused[chunk_id]
            item["semantic_rank"] = rank
            item["semantic_score"] = result.get("score")
            item["rrf_score"] += sw / (self.rrf_k + rank)

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

            item = fused[chunk_id]
            item["bm25_rank"] = rank
            item["bm25_score"] = result.get("score")
            item["rrf_score"] += bw / (self.rrf_k + rank)

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                item["rrf_score"],
                item["semantic_rank"] is not None,
                item["bm25_rank"] is not None,
            ),
            reverse=True,
        )

        for rank, result in enumerate(ranked, start=1):
            result["rank"] = rank
            result["retrieval_sources"] = {
                "semantic": result["semantic_rank"] is not None,
                "bm25": result["bm25_rank"] is not None,
            }

        return ranked

    def _weights_for_query(
        self,
        query: str,
    ) -> QueryProfile | None:
        if not self.adaptive:
            return None

        profile = profile_query(query)

        # The profiler is intentionally conservative. If a future
        # configuration changes the baseline, preserve the constructor
        # weights as the fallback for general queries.
        if profile.query_type == "general":
            semantic = self.semantic_weight
            bm25 = self.bm25_weight
            total = semantic + bm25
            semantic /= total
            bm25 /= total

            return QueryProfile(
                normalized_query=profile.normalized_query,
                token_count=profile.token_count,
                technical_token_count=profile.technical_token_count,
                has_code_identifier=profile.has_code_identifier,
                has_question_word=profile.has_question_word,
                has_error_term=profile.has_error_term,
                has_api_term=profile.has_api_term,
                has_how_to_pattern=profile.has_how_to_pattern,
                query_type=profile.query_type,
                semantic_weight=round(semantic, 4),
                bm25_weight=round(bm25, 4),
            )

        return profile

    def search(self, query: str) -> dict[str, Any]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        profile = self._weights_for_query(query)

        semantic_results = self.semantic.search(
            query,
            top_k=self.semantic_top_k,
        )
        bm25_results = self.bm25.search(
            query,
            top_k=self.bm25_top_k,
        )

        semantic_weight = (
            profile.semantic_weight
            if profile is not None
            else self.semantic_weight
        )
        bm25_weight = (
            profile.bm25_weight
            if profile is not None
            else self.bm25_weight
        )

        hybrid_results = self.reciprocal_rank_fusion(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

        return {
            "query": query,
            "semantic_results": semantic_results,
            "bm25_results": bm25_results,
            "hybrid_results": hybrid_results,
            "query_profile": profile.to_dict() if profile else None,
            "fusion_weights": {
                "semantic": semantic_weight,
                "bm25": bm25_weight,
                "rrf_k": self.rrf_k,
                "adaptive": self.adaptive,
            },
        }


def print_results(
    title: str,
    results: list[dict[str, Any]],
    top_k: int = 10,
) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)

    for result in results[:top_k]:
        metadata = result.get("metadata", {})
        print()
        print(
            f"Rank {result.get('rank', '-')}"
            f" | chunk={result.get('chunk_id', '-')}"
        )
        print(
            f"Score={result.get('score', result.get('rrf_score', 0)):.4f}"
        )
        print(f"Technology={metadata.get('technology', 'unknown')}")
        print(f"Source={metadata.get('source', 'unknown')}")

        section = metadata.get("section", "")
        if section:
            print(f"Section={section}")

        print(f"Text={result.get('text', '')[:400]}...")


def main() -> None:
    print("=" * 80)
    print("HYBRID RAG - ADAPTIVE WEIGHTED RRF RETRIEVAL")
    print("=" * 80)

    retriever = HybridRetriever(
        semantic_top_k=60,
        bm25_top_k=60,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
        adaptive=True,
    )

    while True:
        try:
            query = input(
                "\nEnter query (or type 'exit' to quit): "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        try:
            result = retriever.search(query)

            profile = result.get("query_profile")
            weights = result.get("fusion_weights")

            print()
            print("QUERY PROFILE")
            print("-" * 80)
            print(f"Type          : {profile.get('query_type') if profile else 'baseline'}")
            print(f"Semantic weight: {weights['semantic']:.2f}")
            print(f"BM25 weight    : {weights['bm25']:.2f}")

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
                "ADAPTIVE HYBRID RESULTS",
                result["hybrid_results"],
                top_k=10,
            )

        except Exception as exc:
            print()
            print(f"[ERROR] Retrieval failed: {exc}")


if __name__ == "__main__":
    main()
