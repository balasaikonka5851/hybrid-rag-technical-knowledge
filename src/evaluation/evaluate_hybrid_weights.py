"""
HybridRAG - Weighted RRF Experiment

Evaluates different BM25/Semantic fusion weights using the same
retrieval candidates and the existing evaluation ground truth.

Metrics:
- Hit@1
- Hit@3
- Hit@5
- Hit@10
- MRR

Important:
The candidate retrieval is performed only once per query.
Different fusion weights are then evaluated on the same candidates.
This makes the comparison fair and avoids repeatedly loading models/indexes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------
# Make project root importable when running the file directly
# ---------------------------------------------------------------------

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.bm25 import BM25Retriever, load_chunks
from src.retrieval.semantic import SemanticRetriever

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "queries.json"
)

TOP_K = 20
RRF_K = 60

# Configurations to test.
# Tuple = (semantic_weight, bm25_weight)

WEIGHT_CONFIGURATIONS = [
    (1.00, 0.00),
    (0.75, 0.25),
    (0.50, 0.50),
    (0.25, 0.75),
    (0.00, 1.00),
]


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def load_evaluation_queries() -> list[dict[str, Any]]:
    """Load and validate evaluation queries."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file not found:\n{EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "queries.json must contain a JSON list."
        )

    if not data:
        raise ValueError(
            "queries.json contains no evaluation queries."
        )

    required_fields = {
        "id",
        "query",
        "technology",
        "relevant_sources",
    }

    for index, item in enumerate(data):

        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation query #{index + 1} is not an object."
            )

        missing = required_fields - item.keys()

        if missing:
            raise ValueError(
                f"Evaluation query #{index + 1} is missing: "
                f"{sorted(missing)}"
            )

        if not isinstance(item["relevant_sources"], list):
            raise ValueError(
                f"{item['id']}: relevant_sources must be a list."
            )

        if not item["relevant_sources"]:
            raise ValueError(
                f"{item['id']}: relevant_sources is empty."
            )

    return data


def normalize_source(
    source: str,
    technology: str | None = None,
) -> str:
    """
    Normalize corpus and ground-truth source paths.

    Examples:

        data/raw/python/tutorial/errors.txt
        python/tutorial/errors.txt
        tutorial/errors.txt

    become:

        tutorial/errors.txt
    """

    if not isinstance(source, str):
        return ""

    normalized = source.strip().replace("\\", "/")

    # Remove known raw-data prefix.
    prefixes = (
        "data/raw/",
        "./data/raw/",
        "/data/raw/",
    )

    for prefix in prefixes:
        if normalized.lower().startswith(prefix.lower()):
            normalized = normalized[len(prefix):]
            break

    # Remove technology prefix only when it actually exists.
    if technology:
        technology_prefix = technology.strip().lower() + "/"

        if normalized.lower().startswith(technology_prefix):
            normalized = normalized[len(technology_prefix):]

    return normalized.strip("/").lower()


def source_from_result(
    result: dict[str, Any],
) -> str:
    """Extract source path from a retrieval result."""

    metadata = result.get("metadata", {})

    if not isinstance(metadata, dict):
        return ""

    return str(metadata.get("source", ""))


def result_is_relevant(
    result: dict[str, Any],
    evaluation_item: dict[str, Any],
) -> bool:
    """
    Determine whether a retrieval result matches the ground truth.

    Both technology and source path must match.
    """

    metadata = result.get("metadata", {})

    if not isinstance(metadata, dict):
        return False

    result_technology = str(
        metadata.get("technology", "")
    ).strip().lower()

    expected_technology = str(
        evaluation_item["technology"]
    ).strip().lower()

    if result_technology != expected_technology:
        return False

    result_source = normalize_source(
        source_from_result(result),
        result_technology,
    )

    expected_sources = {
        normalize_source(
            source,
            expected_technology,
        )
        for source in evaluation_item["relevant_sources"]
    }

    return result_source in expected_sources


# ---------------------------------------------------------------------
# Ground-truth validation
# ---------------------------------------------------------------------

def validate_ground_truth(
    queries: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """
    Verify every ground-truth source actually exists in the corpus.
    """

    corpus_sources: set[tuple[str, str]] = set()

    for chunk in chunks:

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            continue

        technology = str(
            metadata.get("technology", "")
        ).strip().lower()

        source = normalize_source(
            str(metadata.get("source", "")),
            technology,
        )

        corpus_sources.add(
            (technology, source)
        )

    missing: list[str] = []

    for item in queries:

        technology = str(
            item["technology"]
        ).strip().lower()

        for source in item["relevant_sources"]:

            normalized = normalize_source(
                source,
                technology,
            )

            if (technology, normalized) not in corpus_sources:

                missing.append(
                    f"{item['id']}: "
                    f"{technology}/{normalized}"
                )

    if missing:

        print()
        print("[ERROR] Ground-truth validation failed.")

        for source in missing:
            print(f"  - {source}")

        raise ValueError(
            f"{len(missing)} ground-truth source(s) "
            "were not found in the corpus."
        )

    print(
        f"[OK] Ground-truth validated against "
        f"{len(chunks):,} corpus chunks."
    )


# ---------------------------------------------------------------------
# Weighted RRF
# ---------------------------------------------------------------------

def weighted_rrf(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    semantic_weight: float,
    bm25_weight: float,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Fuse semantic and BM25 rankings using weighted RRF.

    Formula:

        semantic_weight / (rrf_k + semantic_rank)
        +
        bm25_weight / (rrf_k + bm25_rank)

    Results are merged using chunk_id.
    """

    if semantic_weight < 0:
        raise ValueError(
            "semantic_weight cannot be negative."
        )

    if bm25_weight < 0:
        raise ValueError(
            "bm25_weight cannot be negative."
        )

    if semantic_weight == 0 and bm25_weight == 0:
        raise ValueError(
            "At least one weight must be greater than zero."
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than zero."
        )

    fused: dict[str, dict[str, Any]] = {}

    # -------------------------------------------------------------
    # Semantic rankings
    # -------------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        chunk_id = result.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result.get("text", ""),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "semantic_rank": None,
                "bm25_rank": None,
                "semantic_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["semantic_rank"] = rank
        fused[chunk_id]["semantic_score"] = result.get(
            "score"
        )

        fused[chunk_id]["rrf_score"] += (
            semantic_weight
            / (rrf_k + rank)
        )

    # -------------------------------------------------------------
    # BM25 rankings
    # -------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = result.get("chunk_id")

        if not chunk_id:
            continue

        if chunk_id not in fused:

            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "text": result.get("text", ""),
                "metadata": result.get(
                    "metadata",
                    {},
                ),
                "semantic_rank": None,
                "bm25_rank": None,
                "semantic_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            }

        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["bm25_score"] = result.get(
            "score"
        )

        fused[chunk_id]["rrf_score"] += (
            bm25_weight
            / (rrf_k + rank)
        )

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            item["rrf_score"],
            item["semantic_score"]
            if item["semantic_score"] is not None
            else float("-inf"),
        ),
        reverse=True,
    )

    for rank, result in enumerate(
        ranked,
        start=1,
    ):
        result["rank"] = rank

    return ranked


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(
    results: list[dict[str, Any]],
    evaluation_item: dict[str, Any],
) -> dict[str, float]:

    relevance = [
        result_is_relevant(
            result,
            evaluation_item,
        )
        for result in results[:10]
    ]

    def hit_at(k: int) -> float:

        return float(
            any(relevance[:k])
        )

    first_relevant_rank = None

    for index, is_relevant in enumerate(
        relevance,
        start=1,
    ):

        if is_relevant:
            first_relevant_rank = index
            break

    if first_relevant_rank is None:
        reciprocal_rank = 0.0
    else:
        reciprocal_rank = (
            1.0 / first_relevant_rank
        )

    return {
        "hit@1": hit_at(1),
        "hit@3": hit_at(3),
        "hit@5": hit_at(5),
        "hit@10": hit_at(10),
        "mrr": reciprocal_rank,
        "first_rank": (
            float(first_relevant_rank)
            if first_relevant_rank is not None
            else 0.0
        ),
    }


def aggregate_metrics(
    per_query_metrics: list[dict[str, float]],
) -> dict[str, float]:

    if not per_query_metrics:
        raise ValueError(
            "No query metrics were provided."
        )

    count = len(per_query_metrics)

    return {
        "hit@1": sum(
            item["hit@1"]
            for item in per_query_metrics
        ) / count,

        "hit@3": sum(
            item["hit@3"]
            for item in per_query_metrics
        ) / count,

        "hit@5": sum(
            item["hit@5"]
            for item in per_query_metrics
        ) / count,

        "hit@10": sum(
            item["hit@10"]
            for item in per_query_metrics
        ) / count,

        "mrr": sum(
            item["mrr"]
            for item in per_query_metrics
        ) / count,
    }


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 100)
    print("HYBRID RAG - WEIGHTED RRF EXPERIMENT")
    print("=" * 100)

    # -------------------------------------------------------------
    # Load evaluation set
    # -------------------------------------------------------------

    queries = load_evaluation_queries()

    print()
    print(
        f"[OK] Evaluation queries: {len(queries)}"
    )

    print(
        f"[OK] Evaluation file: {EVALUATION_FILE}"
    )

    # -------------------------------------------------------------
    # Load corpus once
    # -------------------------------------------------------------

    print()
    print("Loading corpus...")

    chunks = load_chunks()

    print(
        f"[OK] Corpus chunks: {len(chunks):,}"
    )

    validate_ground_truth(
        queries,
        chunks,
    )

    # -------------------------------------------------------------
    # Initialize BM25 once
    # -------------------------------------------------------------

    print()
    print("Initializing BM25 once...")

    bm25 = BM25Retriever(chunks)

    print("[OK] BM25 ready.")

    # -------------------------------------------------------------
    # Initialize semantic once
    # -------------------------------------------------------------

    print()
    print("Initializing semantic retrieval once...")

    semantic = SemanticRetriever()

    print("[OK] Semantic retrieval ready.")

    # -------------------------------------------------------------
    # Retrieve candidates ONCE
    # -------------------------------------------------------------

    print()
    print("=" * 100)
    print("COLLECTING RETRIEVAL CANDIDATES")
    print("=" * 100)

    cached_results: list[
        tuple[
            dict[str, Any],
            list[dict[str, Any]],
            list[dict[str, Any]],
        ]
    ] = []

    for index, item in enumerate(
        queries,
        start=1,
    ):

        query_id = item["id"]
        query = item["query"]

        print(
            f"[{index:02d}/{len(queries):02d}] "
            f"{query_id}: {query}"
        )

        semantic_results = semantic.search(
            query,
            top_k=TOP_K,
        )

        bm25_results = bm25.search(
            query,
            top_k=TOP_K,
        )

        cached_results.append(
            (
                item,
                semantic_results,
                bm25_results,
            )
        )

    print()
    print(
        "[OK] Candidate retrieval completed."
    )

    # -------------------------------------------------------------
    # Evaluate configurations
    # -------------------------------------------------------------

    experiment_results: list[
        tuple[
            float,
            float,
            dict[str, float],
            list[dict[str, float]],
        ]
    ] = []

    print()
    print("=" * 100)
    print("TESTING WEIGHT CONFIGURATIONS")
    print("=" * 100)

    for semantic_weight, bm25_weight in (
        WEIGHT_CONFIGURATIONS
    ):

        per_query = []

        for (
            item,
            semantic_results,
            bm25_results,
        ) in cached_results:

            fused = weighted_rrf(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                semantic_weight=semantic_weight,
                bm25_weight=bm25_weight,
                rrf_k=RRF_K,
            )

            metrics = calculate_metrics(
                fused,
                item,
            )

            per_query.append(metrics)

        aggregate = aggregate_metrics(
            per_query
        )

        experiment_results.append(
            (
                semantic_weight,
                bm25_weight,
                aggregate,
                per_query,
            )
        )

        print(
            f"Semantic={semantic_weight:.2f} | "
            f"BM25={bm25_weight:.2f} | "
            f"MRR={aggregate['mrr']:.3f} | "
            f"Hit@1={aggregate['hit@1']:.3f} | "
            f"Hit@5={aggregate['hit@5']:.3f}"
        )

    # -------------------------------------------------------------
    # Comparison table
    # -------------------------------------------------------------

    print()
    print("=" * 100)
    print("WEIGHTED RRF COMPARISON")
    print("=" * 100)

    print(
        f"{'Semantic':>10} "
        f"{'BM25':>10} "
        f"{'Hit@1':>10} "
        f"{'Hit@3':>10} "
        f"{'Hit@5':>10} "
        f"{'Hit@10':>10} "
        f"{'MRR':>10}"
    )

    print("-" * 75)

    for (
        semantic_weight,
        bm25_weight,
        metrics,
        _,
    ) in experiment_results:

        print(
            f"{semantic_weight:>10.2f} "
            f"{bm25_weight:>10.2f} "
            f"{metrics['hit@1']:>10.3f} "
            f"{metrics['hit@3']:>10.3f} "
            f"{metrics['hit@5']:>10.3f} "
            f"{metrics['hit@10']:>10.3f} "
            f"{metrics['mrr']:>10.3f}"
        )

    # -------------------------------------------------------------
    # Select best configuration by MRR
    # -------------------------------------------------------------

    best = max(
        experiment_results,
        key=lambda item: (
            item[2]["mrr"],
            item[2]["hit@5"],
            item[2]["hit@1"],
        ),
    )

    (
        best_semantic_weight,
        best_bm25_weight,
        best_metrics,
        _,
    ) = best

    print()
    print("=" * 100)
    print("BEST CONFIGURATION")
    print("=" * 100)

    print(
        f"Semantic weight : {best_semantic_weight:.2f}"
    )

    print(
        f"BM25 weight     : {best_bm25_weight:.2f}"
    )

    print(
        f"Hit@1           : {best_metrics['hit@1']:.3f}"
    )

    print(
        f"Hit@3           : {best_metrics['hit@3']:.3f}"
    )

    print(
        f"Hit@5           : {best_metrics['hit@5']:.3f}"
    )

    print(
        f"Hit@10          : {best_metrics['hit@10']:.3f}"
    )

    print(
        f"MRR             : {best_metrics['mrr']:.3f}"
    )

    print()
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()