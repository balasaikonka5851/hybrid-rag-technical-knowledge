from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid import HybridRetriever


QUERIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "queries.json"
)

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_DEPTH = 60
OUTPUT_K = 10

BATCH_SIZE = 16
MAX_LENGTH = 512

SEMANTIC_WEIGHT = 0.75
BM25_WEIGHT = 0.25
RRF_K = 60


def load_queries() -> list[dict[str, Any]]:
    if not QUERIES_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found: {QUERIES_PATH}"
        )

    with QUERIES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    if not isinstance(queries, list):
        raise ValueError(
            "Evaluation queries must be a list."
        )

    if not queries:
        raise ValueError(
            "Evaluation query list is empty."
        )

    return queries


def normalize_source(source: str) -> str:
    return source.replace("\\", "/").strip()


def get_relevant_sources(
    item: dict[str, Any],
) -> set[str]:

    sources = item.get(
        "relevant_sources",
        [],
    )

    if isinstance(sources, str):
        sources = [sources]

    return {
        normalize_source(str(source))
        for source in sources
    }


def is_relevant(
    result: dict[str, Any],
    relevant_sources: set[str],
    technology: str | None,
) -> bool:

    metadata = result.get(
        "metadata",
        {},
    )

    result_source = normalize_source(
        str(
            metadata.get(
                "source",
                "",
            )
        )
    )

    for target in relevant_sources:

        if result_source.endswith(target):
            return True

        if technology:

            expected = (
                f"data/raw/"
                f"{technology}/"
                f"{target}"
            )

            if result_source.endswith(expected):
                return True

    return False


def reciprocal_rank(
    results: list[dict[str, Any]],
    relevant_sources: set[str],
    technology: str | None,
) -> float:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_relevant(
            result,
            relevant_sources,
            technology,
        ):
            return 1.0 / rank

    return 0.0


def hit_at_k(
    results: list[dict[str, Any]],
    relevant_sources: set[str],
    technology: str | None,
    k: int,
) -> float:

    for result in results[:k]:

        if is_relevant(
            result,
            relevant_sources,
            technology,
        ):
            return 1.0

    return 0.0


def evaluate() -> None:

    print("=" * 100)
    print("HYBRID RAG - RERANKER DEPTH EVALUATION")
    print("=" * 100)

    queries = load_queries()

    print(
        f"[OK] Evaluation queries: {len(queries)}"
    )

    print(
        f"[OK] Candidate depth: {CANDIDATE_DEPTH}"
    )

    print(
        f"[OK] Output K: {OUTPUT_K}"
    )

    print(
        f"[OK] Reranker: {MODEL_NAME}"
    )

    print()
    print("Hybrid configuration:")
    print(
        f"  Semantic weight : "
        f"{SEMANTIC_WEIGHT}"
    )
    print(
        f"  BM25 weight     : "
        f"{BM25_WEIGHT}"
    )
    print(
        f"  RRF k           : {RRF_K}"
    )

    # ---------------------------------------------------------
    # Hybrid retriever
    # ---------------------------------------------------------

    print()
    print("Loading hybrid retriever...")

    retriever = HybridRetriever(
        semantic_top_k=CANDIDATE_DEPTH,
        bm25_top_k=CANDIDATE_DEPTH,
        semantic_weight=SEMANTIC_WEIGHT,
        bm25_weight=BM25_WEIGHT,
        rrf_k=RRF_K,
    )

    print(
        "[OK] Hybrid retriever loaded."
    )

    # ---------------------------------------------------------
    # Cross encoder
    # ---------------------------------------------------------

    print()
    print("Loading cross-encoder...")

    reranker = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
    )

    print(
        "[OK] Cross-encoder loaded."
    )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    totals = {
        "hybrid": {
            1: 0.0,
            3: 0.0,
            5: 0.0,
            10: 0.0,
            "mrr": 0.0,
        },
        "reranked": {
            1: 0.0,
            3: 0.0,
            5: 0.0,
            10: 0.0,
            "mrr": 0.0,
        },
    }

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    for index, item in enumerate(
        queries,
        start=1,
    ):

        query_id = item.get(
            "id",
            f"q{index}",
        )

        query = str(
            item.get(
                "query",
                "",
            )
        ).strip()

        if not query:
            raise ValueError(
                f"Query {query_id} is empty."
            )

        relevant_sources = (
            get_relevant_sources(item)
        )

        if not relevant_sources:
            raise ValueError(
                f"Query {query_id} has no "
                "relevant_sources."
            )

        technology = item.get(
            "technology"
        )

        # -----------------------------------------------------
        # IMPORTANT:
        # HybridRetriever.search() returns a dictionary.
        # Candidate depth is configured in the constructor.
        # -----------------------------------------------------

        retrieval = retriever.search(
            query
        )

        candidates = retrieval[
            "hybrid_results"
        ]

        hybrid_results = candidates[
            :OUTPUT_K
        ]

        # -----------------------------------------------------
        # Cross-encoder scoring
        # -----------------------------------------------------

        if candidates:

            pairs = [
                [
                    query,
                    str(
                        result.get(
                            "text",
                            "",
                        )
                    ),
                ]
                for result in candidates
            ]

            scores = reranker.predict(
                pairs,
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
            )

            if len(scores) != len(
                candidates
            ):
                raise RuntimeError(
                    "Reranker returned an "
                    "unexpected number of scores."
                )

            scored = []

            for result, score in zip(
                candidates,
                scores,
            ):

                item_result = dict(
                    result
                )

                item_result[
                    "reranker_score"
                ] = float(score)

                scored.append(
                    item_result
                )

            scored.sort(
                key=lambda result:
                result["reranker_score"],
                reverse=True,
            )

            reranked_results = (
                scored[:OUTPUT_K]
            )

        else:
            reranked_results = []

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        hybrid_rr = reciprocal_rank(
            hybrid_results,
            relevant_sources,
            technology,
        )

        reranked_rr = reciprocal_rank(
            reranked_results,
            relevant_sources,
            technology,
        )

        for k in (
            1,
            3,
            5,
            10,
        ):

            totals["hybrid"][k] += (
                hit_at_k(
                    hybrid_results,
                    relevant_sources,
                    technology,
                    k,
                )
            )

            totals["reranked"][k] += (
                hit_at_k(
                    reranked_results,
                    relevant_sources,
                    technology,
                    k,
                )
            )

        totals["hybrid"]["mrr"] += (
            hybrid_rr
        )

        totals["reranked"]["mrr"] += (
            reranked_rr
        )

        print(
            f"[{index:02d}/{len(queries)}] "
            f"{query_id:>4} | "
            f"candidates={len(candidates):2d} | "
            f"HybridRR={hybrid_rr:.3f} | "
            f"RerankRR={reranked_rr:.3f}"
        )

    # ---------------------------------------------------------
    # Final results
    # ---------------------------------------------------------

    n = len(queries)

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)

    print(
        f"{'Method':<30}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 80)

    for method, label in (
        (
            "hybrid",
            "Hybrid @ 60",
        ),
        (
            "reranked",
            "Hybrid @ 60 + Reranker",
        ),
    ):

        values = totals[method]

        print(
            f"{label:<30}"
            f"{values[1] / n:>10.3f}"
            f"{values[3] / n:>10.3f}"
            f"{values[5] / n:>10.3f}"
            f"{values[10] / n:>10.3f}"
            f"{values['mrr'] / n:>10.3f}"
        )

    print("=" * 100)

    hybrid_mrr = (
        totals["hybrid"]["mrr"] / n
    )

    reranked_mrr = (
        totals["reranked"]["mrr"] / n
    )

    change = (
        reranked_mrr
        - hybrid_mrr
    )

    print()
    print(
        f"MRR change: {change:+.3f}"
    )

    if reranked_mrr > hybrid_mrr:

        print(
            "[WINNER] Reranking improves MRR."
        )

    elif reranked_mrr < hybrid_mrr:

        print(
            "[RESULT] Reranking decreases MRR."
        )

    else:

        print(
            "[RESULT] Reranking produces "
            "the same MRR."
        )

    print()
    print(
        "[OK] Reranker depth evaluation "
        "completed."
    )


if __name__ == "__main__":
    evaluate()