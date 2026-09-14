from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from src.retrieval.bm25 import BM25Retriever, load_chunks
from src.retrieval.semantic import SemanticRetriever
from src.retrieval.hybrid import HybridRetriever


# ============================================================================
# CONFIGURATION
# ============================================================================

EVALUATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "queries.json"
)

TOP_K_VALUES = (1, 3, 5, 10)

MAX_RESULTS = max(TOP_K_VALUES)

SEMANTIC_CANDIDATES = 20
BM25_CANDIDATES = 20


# ============================================================================
# DATA LOADING
# ============================================================================

def load_evaluation_queries() -> list[dict[str, Any]]:
    """Load and validate the evaluation dataset."""

    if not EVALUATION_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found:\n{EVALUATION_FILE}"
        )

    try:
        with EVALUATION_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            queries = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in evaluation file:\n"
            f"{EVALUATION_FILE}\n\n"
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(queries, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON array."
        )

    if not queries:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    required_fields = {
        "id",
        "query",
        "technology",
        "relevant_sources",
    }

    seen_ids: set[str] = set()

    for index, item in enumerate(queries, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Evaluation item #{index} must be a JSON object."
            )

        missing = required_fields - set(item.keys())

        if missing:
            raise ValueError(
                f"Evaluation item #{index} is missing fields: "
                f"{sorted(missing)}"
            )

        query_id = str(item["id"]).strip()

        if not query_id:
            raise ValueError(
                f"Evaluation item #{index} has an empty id."
            )

        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate evaluation query id: {query_id}"
            )

        seen_ids.add(query_id)

        query = item["query"]

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"Query {query_id} has an invalid query."
            )

        technology = item["technology"]

        if not isinstance(technology, str):
            raise ValueError(
                f"Query {query_id} has an invalid technology."
            )

        technology = technology.strip().lower()

        if technology not in {"python", "fastapi"}:
            raise ValueError(
                f"Query {query_id} has unsupported technology "
                f"'{technology}'. Expected 'python' or 'fastapi'."
            )

        relevant_sources = item["relevant_sources"]

        if not isinstance(relevant_sources, list):
            raise ValueError(
                f"Query {query_id}: relevant_sources must be a list."
            )

        if not relevant_sources:
            raise ValueError(
                f"Query {query_id}: relevant_sources cannot be empty."
            )

        for source in relevant_sources:
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    f"Query {query_id} contains an invalid relevant source."
                )

    return queries


# ============================================================================
# SOURCE NORMALIZATION
# ============================================================================

def normalize_source(
    source: str,
    technology: str | None = None,
) -> str:
    """
    Normalize documentation source paths.

    Examples:

        data/raw/fastapi/tutorial/handling-errors.md
        -> tutorial/handling-errors.md

        data/raw/python/tutorial/errors.txt
        -> tutorial/errors.txt

        tutorial/errors.txt
        -> tutorial/errors.txt

    We only remove a technology prefix when it is actually present.
    """

    value = str(source).replace("\\", "/").strip()

    # Remove leading "./"
    while value.startswith("./"):
        value = value[2:]

    # Locate data/raw anywhere in an absolute Windows/Linux path.
    marker = "data/raw/"

    if marker in value:
        value = value.split(marker, 1)[1]

    # Remove the known technology directory.
    normalized_technology = (
        technology.strip().lower()
        if isinstance(technology, str)
        else None
    )

    if normalized_technology:
        prefix = normalized_technology + "/"

        if value.lower().startswith(prefix):
            value = value[len(prefix):]

    else:
        # Only remove these directories when explicitly present.
        for known_technology in ("fastapi", "python"):
            prefix = known_technology + "/"

            if value.lower().startswith(prefix):
                value = value[len(prefix):]
                break

    return value.strip("/")


def build_relevant_sources(
    query: dict[str, Any],
) -> set[str]:
    """Return normalized ground-truth source paths."""

    technology = query["technology"]

    return {
        normalize_source(
            source,
            technology=technology,
        )
        for source in query["relevant_sources"]
    }


def result_source(
    result: dict[str, Any],
) -> str:
    """Extract and normalize a retrieved result's source path."""

    metadata = result.get("metadata", {})

    if not isinstance(metadata, dict):
        return ""

    source = metadata.get("source", "")

    technology = metadata.get("technology")

    return normalize_source(
        str(source),
        technology=str(technology)
        if technology is not None
        else None,
    )


# ============================================================================
# GROUND-TRUTH VALIDATION
# ============================================================================

def validate_ground_truth(
    queries: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """
    Verify that every evaluation source actually exists in the corpus.

    This prevents invalid ground-truth paths from silently producing
    misleading evaluation scores.
    """

    corpus_sources_by_technology: dict[str, set[str]] = {}

    for chunk in chunks:

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            continue

        technology = str(
            metadata.get("technology", "")
        ).strip().lower()

        source = metadata.get("source", "")

        if not technology or not source:
            continue

        normalized = normalize_source(
            str(source),
            technology=technology,
        )

        corpus_sources_by_technology.setdefault(
            technology,
            set(),
        ).add(normalized)

    errors: list[str] = []

    for query in queries:

        query_id = query["id"]
        technology = query["technology"].strip().lower()

        available_sources = corpus_sources_by_technology.get(
            technology,
            set(),
        )

        for expected_source in build_relevant_sources(query):

            if expected_source not in available_sources:
                errors.append(
                    f"{query_id}: source not found in corpus -> "
                    f"{technology}/{expected_source}"
                )

    if errors:

        message = (
            "\nGround-truth validation failed.\n"
            "The following evaluation sources do not exist "
            "in the processed corpus:\n\n"
            + "\n".join(
                f"  - {error}"
                for error in errors
            )
        )

        raise ValueError(message)

    print(
        f"[OK] Ground-truth validated against "
        f"{len(chunks):,} corpus chunks."
    )


# ============================================================================
# RELEVANCE
# ============================================================================

def is_relevant(
    result: dict[str, Any],
    query: dict[str, Any],
) -> bool:
    """
    Determine whether a retrieved result belongs to one of the
    ground-truth relevant documentation sources.

    Technology is also checked to prevent a same-named path in another
    corpus from being counted as relevant.
    """

    metadata = result.get("metadata", {})

    if not isinstance(metadata, dict):
        return False

    result_technology = str(
        metadata.get("technology", "")
    ).strip().lower()

    query_technology = str(
        query["technology"]
    ).strip().lower()

    if result_technology != query_technology:
        return False

    source = result_source(result)

    relevant_sources = build_relevant_sources(query)

    return source in relevant_sources


# ============================================================================
# METRICS
# ============================================================================

def hit_at_k(
    results: list[dict[str, Any]],
    query: dict[str, Any],
    k: int,
) -> float:
    """Return 1 when a relevant result appears within top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    for result in results[:k]:

        if is_relevant(result, query):
            return 1.0

    return 0.0


def reciprocal_rank(
    results: list[dict[str, Any]],
    query: dict[str, Any],
) -> float:
    """
    Reciprocal rank of the first relevant result.

        rank 1 -> 1.0
        rank 2 -> 0.5
        rank 3 -> 0.333...
        no result -> 0.0
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if is_relevant(result, query):
            return 1.0 / rank

    return 0.0


def calculate_metrics(
    all_results: list[list[dict[str, Any]]],
    queries: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate aggregate retrieval metrics."""

    if len(all_results) != len(queries):
        raise ValueError(
            "Number of result sets does not match number of queries."
        )

    count = len(queries)

    if count == 0:
        raise ValueError(
            "Cannot calculate metrics for zero queries."
        )

    metrics: dict[str, float] = {}

    for k in TOP_K_VALUES:

        hits = [
            hit_at_k(
                results,
                query,
                k,
            )
            for results, query in zip(
                all_results,
                queries,
            )
        ]

        metrics[f"Hit@{k}"] = sum(hits) / count

    reciprocal_ranks = [
        reciprocal_rank(
            results,
            query,
        )
        for results, query in zip(
            all_results,
            queries,
        )
    ]

    metrics["MRR"] = (
        sum(reciprocal_ranks) / count
    )

    return metrics


# ============================================================================
# RETRIEVAL EXECUTION
# ============================================================================

def run_bm25(
    retriever: BM25Retriever,
    query: str,
) -> list[dict[str, Any]]:
    """Run BM25 and return at most MAX_RESULTS results."""

    results = retriever.search(
        query,
        top_k=MAX_RESULTS,
    )

    if not isinstance(results, list):
        raise TypeError(
            "BM25Retriever.search() must return a list."
        )

    return results[:MAX_RESULTS]


def run_semantic(
    retriever: SemanticRetriever,
    query: str,
) -> list[dict[str, Any]]:
    """Run semantic retrieval."""

    results = retriever.search(
        query,
        top_k=MAX_RESULTS,
    )

    if not isinstance(results, list):
        raise TypeError(
            "SemanticRetriever.search() must return a list."
        )

    return results[:MAX_RESULTS]


def run_hybrid(
    retriever: HybridRetriever,
    query: str,
) -> list[dict[str, Any]]:
    """
    Run hybrid retrieval.

    Candidate counts are configured when HybridRetriever is initialized.
    Therefore they must NOT be passed to search().
    """

    result = retriever.search(query)

    if not isinstance(result, dict):
        raise TypeError(
            "HybridRetriever.search() must return a dictionary."
        )

    hybrid_results = result.get(
        "hybrid_results"
    )

    if not isinstance(
        hybrid_results,
        list,
    ):
        raise ValueError(
            "Hybrid retrieval did not return "
            "a valid 'hybrid_results' list."
        )

    return hybrid_results[:MAX_RESULTS]


# ============================================================================
# PER-METHOD EVALUATION
# ============================================================================

RetrievalFunction = Callable[
    [str],
    list[dict[str, Any]],
]


def evaluate_method(
    method_name: str,
    retrieval_function: RetrievalFunction,
    queries: list[dict[str, Any]],
) -> tuple[
    list[list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Evaluate one retrieval method."""

    all_results: list[
        list[dict[str, Any]]
    ] = []

    detailed_results: list[
        dict[str, Any]
    ] = []

    print()
    print("=" * 90)
    print(f"{method_name.upper()} EVALUATION")
    print("=" * 90)

    for query in queries:

        query_id = query["id"]
        query_text = query["query"]

        results = retrieval_function(
            query_text
        )

        all_results.append(results)

        rr = reciprocal_rank(
            results,
            query,
        )

        first_relevant_rank: int | None = None

        for rank, result in enumerate(
            results,
            start=1,
        ):

            if is_relevant(
                result,
                query,
            ):
                first_relevant_rank = rank
                break

        hits = {
            f"Hit@{k}": hit_at_k(
                results,
                query,
                k,
            )
            for k in TOP_K_VALUES
        }

        detailed_results.append(
            {
                "id": query_id,
                "query": query_text,
                "technology": query["technology"],
                "first_relevant_rank": (
                    first_relevant_rank
                ),
                "reciprocal_rank": rr,
                **hits,
            }
        )

        rank_display = (
            str(first_relevant_rank)
            if first_relevant_rank is not None
            else "-"
        )

        print(
            f"{query_id:>4} | "
            f"first relevant rank={rank_display:>2} | "
            f"RR={rr:.3f}"
        )

    return (
        all_results,
        detailed_results,
    )


# ============================================================================
# COMPARISON
# ============================================================================

def print_comparison(
    metric_results: dict[
        str,
        dict[str, float],
    ],
) -> None:
    """Print aggregate metric comparison."""

    print()
    print("=" * 90)
    print("RETRIEVAL METHOD COMPARISON")
    print("=" * 90)

    header = (
        f"{'Method':<12}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print(header)
    print("-" * len(header))

    for method, metrics in metric_results.items():

        print(
            f"{method:<12}"
            f"{metrics['Hit@1']:>10.3f}"
            f"{metrics['Hit@3']:>10.3f}"
            f"{metrics['Hit@5']:>10.3f}"
            f"{metrics['Hit@10']:>10.3f}"
            f"{metrics['MRR']:>10.3f}"
        )

    print()


# ============================================================================
# DETAILED SUMMARY
# ============================================================================

def print_detailed_results(
    method_details: dict[
        str,
        list[dict[str, Any]],
    ],
) -> None:
    """Print per-query metric results."""

    print("=" * 90)
    print("DETAILED RESULTS")
    print("=" * 90)

    for method, details in method_details.items():

        print()
        print(method)
        print("-" * 90)

        for item in details:

            rank = item["first_relevant_rank"]

            rank_display = (
                str(rank)
                if rank is not None
                else "-"
            )

            print(
                f"{item['id']} | "
                f"Hit@1={item['Hit@1']:.0f} | "
                f"Hit@3={item['Hit@3']:.0f} | "
                f"Hit@5={item['Hit@5']:.0f} | "
                f"Hit@10={item['Hit@10']:.0f} | "
                f"First={rank_display} | "
                f"RR={item['reciprocal_rank']:.3f}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 90)
    print("HYBRID RAG - RETRIEVAL EVALUATION")
    print("=" * 90)

    # ------------------------------------------------------------------------
    # Load evaluation data
    # ------------------------------------------------------------------------

    queries = load_evaluation_queries()

    print()
    print(
        f"[OK] Evaluation queries: {len(queries)}"
    )

    print(
        f"[OK] Evaluation file: "
        f"{EVALUATION_FILE}"
    )

    print(
        "[OK] Metrics: "
        "Hit@1, Hit@3, Hit@5, Hit@10, MRR"
    )

    # ------------------------------------------------------------------------
    # Load corpus once
    # ------------------------------------------------------------------------

    print()
    print("Loading processed corpus...")

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError(
            "Processed corpus contains zero chunks."
        )

    print(
        f"[OK] Corpus chunks: {len(chunks):,}"
    )

    # ------------------------------------------------------------------------
    # Validate ground truth BEFORE retrieval
    # ------------------------------------------------------------------------

    print()
    print("Validating evaluation ground truth...")

    validate_ground_truth(
        queries,
        chunks,
    )

    # ------------------------------------------------------------------------
    # Initialize BM25
    # ------------------------------------------------------------------------

    print()
    print("[1/3] Initializing BM25...")

    bm25 = BM25Retriever(
        chunks
    )

    print(
        "[OK] BM25 ready."
    )

    # ------------------------------------------------------------------------
    # Initialize semantic
    # ------------------------------------------------------------------------

    print()
    print("[2/3] Initializing semantic retrieval...")

    semantic = SemanticRetriever()

    print(
        "[OK] Semantic retrieval ready."
    )

    # ------------------------------------------------------------------------
    # Initialize hybrid
    # ------------------------------------------------------------------------

    print()
    print("[3/3] Initializing hybrid retrieval...")

    hybrid = HybridRetriever(
        semantic_top_k=SEMANTIC_CANDIDATES,
        bm25_top_k=BM25_CANDIDATES,
    )

    print(
        "[OK] Hybrid retrieval ready."
    )

    print()
    print(
        "[OK] All retrieval systems initialized."
    )

    # ------------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------------

    bm25_results, bm25_details = evaluate_method(
        "BM25",
        lambda query: run_bm25(
            bm25,
            query,
        ),
        queries,
    )

    semantic_results, semantic_details = evaluate_method(
        "Semantic",
        lambda query: run_semantic(
            semantic,
            query,
        ),
        queries,
    )

    hybrid_results, hybrid_details = evaluate_method(
        "Hybrid",
        lambda query: run_hybrid(
            hybrid,
            query,
        ),
        queries,
    )

    # ------------------------------------------------------------------------
    # Calculate metrics
    # ------------------------------------------------------------------------

    metric_results = {
        "BM25": calculate_metrics(
            bm25_results,
            queries,
        ),
        "Semantic": calculate_metrics(
            semantic_results,
            queries,
        ),
        "Hybrid": calculate_metrics(
            hybrid_results,
            queries,
        ),
    }

    # ------------------------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------------------------

    print_comparison(
        metric_results
    )

    # ------------------------------------------------------------------------
    # Print detailed results
    # ------------------------------------------------------------------------

    print_detailed_results(
        {
            "BM25": bm25_details,
            "Semantic": semantic_details,
            "Hybrid": hybrid_details,
        }
    )

    # ------------------------------------------------------------------------
    # Final sanity checks
    # ------------------------------------------------------------------------

    for method, metrics in metric_results.items():

        for metric_name, value in metrics.items():

            if not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Invalid metric value: "
                    f"{method} {metric_name}={value}"
                )

    print()
    print("=" * 90)
    print("EVALUATION COMPLETE")
    print("=" * 90)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()