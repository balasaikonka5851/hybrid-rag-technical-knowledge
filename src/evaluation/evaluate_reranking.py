from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# ============================================================================
# PROJECT SETUP
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# IMPORTS
# ============================================================================

from src.retrieval.bm25 import BM25Retriever, load_chunks
from src.retrieval.semantic import SemanticRetriever
from src.reranking.reranker import CrossEncoderReranker


# ============================================================================
# PATHS
# ============================================================================

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

EVALUATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "queries.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# First-stage retrieval.
SEMANTIC_CANDIDATES = 20
BM25_CANDIDATES = 20

# Hybrid RRF.
SEMANTIC_WEIGHT = 0.75
BM25_WEIGHT = 0.25
RRF_K = 60

# Cross-encoder.
RERANK_CANDIDATES = 20
RERANK_OUTPUT_K = 20

RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

RERANKER_MAX_LENGTH = 512
RERANKER_BATCH_SIZE = 16

# Console display.
DISPLAY_TOP_K = 5


# ============================================================================
# PATH NORMALIZATION
# ============================================================================

def normalize_path(value: Any) -> str:
    """
    Normalize a source path for reliable comparison.

    Handles:
        C:\\project\\file.md
        C:/project/file.md
        ./file.md
        /file.md
        data/raw/file.md
        repository-relative paths
    """

    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    text = text.replace("\\", "/")

    # Collapse duplicate separators.
    while "//" in text:
        text = text.replace("//", "/")

    # Normalize ./.
    while text.startswith("./"):
        text = text[2:]

    # Remove leading slash.
    text = text.lstrip("/")

    # Project-root normalization.
    project_root = (
        str(PROJECT_ROOT)
        .replace("\\", "/")
        .rstrip("/")
    )

    if text.lower().startswith(
        project_root.lower() + "/"
    ):
        text = text[
            len(project_root) + 1:
        ]

    # Remove known storage prefixes.
    prefixes = (
        "data/raw/",
        "data/processed/",
    )

    changed = True

    while changed:

        changed = False

        lowered = text.lower()

        for prefix in prefixes:

            if lowered.startswith(prefix):

                text = text[
                    len(prefix):
                ]

                changed = True
                break

    return text.strip("/").lower()


# ============================================================================
# SOURCE ALIASES
# ============================================================================

def get_source_aliases(
    source: Any,
    technology: Any = None,
) -> set[str]:
    """
    Generate safe equivalent representations.

    Example:

        fastapi/tutorial/foo.md

    becomes:

        fastapi/tutorial/foo.md
        tutorial/foo.md
    """

    normalized = normalize_path(source)

    if not normalized:
        return set()

    aliases = {
        normalized
    }

    tech = normalize_path(
        technology
    )

    if tech:

        prefix = (
            tech.rstrip("/")
            + "/"
        )

        if normalized.startswith(prefix):

            relative = normalized[
                len(prefix):
            ]

            if relative:
                aliases.add(
                    relative
                )

    return aliases


def sources_match(
    expected_source: Any,
    expected_technology: Any,
    actual_source: Any,
    actual_technology: Any,
) -> bool:
    """
    Safely determine whether two source identifiers
    represent the same document.
    """

    expected_tech = normalize_path(
        expected_technology
    )

    actual_tech = normalize_path(
        actual_technology
    )

    # If both technologies are available,
    # they must agree.
    if (
        expected_tech
        and actual_tech
        and expected_tech != actual_tech
    ):
        return False

    expected_aliases = get_source_aliases(
        expected_source,
        expected_technology,
    )

    actual_aliases = get_source_aliases(
        actual_source,
        actual_technology,
    )

    if expected_aliases & actual_aliases:
        return True

    # Safe suffix comparison.
    for expected in expected_aliases:

        for actual in actual_aliases:

            if (
                actual.endswith(
                    "/" + expected
                )
                or expected.endswith(
                    "/" + actual
                )
            ):
                return True

    return False


# ============================================================================
# EVALUATION DATASET
# ============================================================================

def load_evaluation_queries() -> list[dict[str, Any]]:
    """Load and validate queries.json."""

    if not EVALUATION_PATH.exists():

        raise FileNotFoundError(
            "Evaluation file not found:\n"
            f"{EVALUATION_PATH}"
        )

    try:

        import json

        with EVALUATION_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except Exception as exc:

        raise ValueError(
            "Unable to load evaluation JSON:\n"
            f"{EVALUATION_PATH}\n"
            f"{exc}"
        ) from exc

    if not isinstance(
        data,
        list,
    ):

        raise ValueError(
            "queries.json must contain a JSON list."
        )

    required = {
        "id",
        "query",
        "technology",
        "relevant_sources",
    }

    for index, item in enumerate(
        data,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):

            raise ValueError(
                f"Evaluation item {index} "
                "is not an object."
            )

        missing = (
            required
            - set(item.keys())
        )

        if missing:

            raise ValueError(
                f"Evaluation item {index} "
                f"missing: {sorted(missing)}"
            )

        if not item[
            "relevant_sources"
        ]:

            raise ValueError(
                f"{item['id']} has no "
                "relevant_sources."
            )

    return data


# ============================================================================
# CORPUS VALIDATION
# ============================================================================

def validate_corpus(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Validate only fields actually required by retrieval/evaluation.

    IMPORTANT:
    Raw chunks are NOT required to have a top-level chunk_id.
    """

    if not chunks:

        raise ValueError(
            "Corpus contains zero chunks."
        )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):

        if not isinstance(
            chunk,
            dict,
        ):

            raise ValueError(
                f"Chunk {index} is not a JSON object."
            )

        if "text" not in chunk:

            raise ValueError(
                f"Chunk {index} has no text field."
            )

        if not isinstance(
            chunk["text"],
            str,
        ):

            raise ValueError(
                f"Chunk {index} text is not a string."
            )

        if "metadata" not in chunk:

            raise ValueError(
                f"Chunk {index} has no metadata field."
            )

        if not isinstance(
            chunk["metadata"],
            dict,
        ):

            raise ValueError(
                f"Chunk {index} metadata is not an object."
            )

        metadata = chunk[
            "metadata"
        ]

        if not metadata.get(
            "source"
        ):

            raise ValueError(
                f"Chunk {index} metadata has no source."
            )

    print(
        "[OK] Corpus structure validated."
    )


# ============================================================================
# GROUND TRUTH VALIDATION
# ============================================================================

def validate_ground_truth(
    chunks: list[dict[str, Any]],
    queries: list[dict[str, Any]],
) -> None:
    """
    Confirm that every ground-truth source exists
    in the corpus.

    Matching is based on technology + normalized
    source aliases.
    """

    print()
    print("=" * 100)
    print("VALIDATING GROUND TRUTH")
    print("=" * 100)

    corpus_documents = []

    for chunk in chunks:

        metadata = chunk[
            "metadata"
        ]

        corpus_documents.append(
            {
                "technology": metadata.get(
                    "technology",
                    "",
                ),
                "source": metadata.get(
                    "source",
                    "",
                ),
            }
        )

    expected_count = 0
    matched_count = 0

    for query_item in queries:

        query_id = query_item[
            "id"
        ]

        technology = query_item[
            "technology"
        ]

        for expected_source in query_item[
            "relevant_sources"
        ]:

            expected_count += 1

            matched = False

            for document in corpus_documents:

                if sources_match(
                    expected_source,
                    technology,
                    document["source"],
                    document["technology"],
                ):

                    matched = True
                    break

            if matched:

                matched_count += 1

            else:

                print()
                print(
                    "[ERROR] Ground-truth source "
                    "not found:"
                )

                print(
                    f"  Query      : {query_id}"
                )

                print(
                    f"  Technology : {technology}"
                )

                print(
                    f"  Expected   : {expected_source}"
                )

                # Show same-filename candidates.
                expected_filename = (
                    Path(
                        str(
                            expected_source
                        )
                    )
                    .name
                    .lower()
                )

                candidates = []

                for document in corpus_documents:

                    actual_filename = (
                        Path(
                            str(
                                document[
                                    "source"
                                ]
                            )
                        )
                        .name
                        .lower()
                    )

                    if (
                        actual_filename
                        == expected_filename
                    ):

                        candidates.append(
                            document
                        )

                if candidates:

                    print(
                        "  Possible matches:"
                    )

                    for candidate in candidates[:5]:

                        print(
                            "    "
                            f"{candidate['technology']}/"
                            f"{candidate['source']}"
                        )

    print()
    print(
        f"[OK] Ground-truth matched: "
        f"{matched_count}/{expected_count}"
    )

    if matched_count != expected_count:

        raise ValueError(
            "Ground-truth validation failed."
        )

    print(
        "[OK] Every evaluation source "
        "exists in the corpus."
    )


# ============================================================================
# RESULT ID
# ============================================================================

def get_result_id(
    result: dict[str, Any],
) -> str:
    """
    Retrieve a stable result identifier.

    Retrieval modules normally expose chunk_id.
    This fallback makes the evaluator resilient.
    """

    for key in (
        "chunk_id",
        "id",
    ):

        value = result.get(
            key
        )

        if value is not None:

            return str(value)

    metadata = result.get(
        "metadata",
        {},
    )

    for key in (
        "chunk_id",
        "id",
    ):

        value = metadata.get(
            key
        )

        if value is not None:

            return str(value)

    # Last-resort deterministic identity.
    return (
        normalize_path(
            metadata.get(
                "technology",
                "",
            )
        )
        + "::"
        + normalize_path(
            metadata.get(
                "source",
                "",
            )
        )
        + "::"
        + str(
            result.get(
                "text",
                "",
            )
        )[:200]
    )


# ============================================================================
# RELEVANCE
# ============================================================================

def result_is_relevant(
    result: dict[str, Any],
    query_item: dict[str, Any],
) -> bool:

    metadata = result.get(
        "metadata",
        {},
    )

    actual_source = metadata.get(
        "source",
        "",
    )

    actual_technology = metadata.get(
        "technology",
        "",
    )

    expected_technology = query_item[
        "technology"
    ]

    for expected_source in query_item[
        "relevant_sources"
    ]:

        if sources_match(
            expected_source,
            expected_technology,
            actual_source,
            actual_technology,
        ):

            return True

    return False


# ============================================================================
# METRICS
# ============================================================================

def hit_at_k(
    results: list[dict[str, Any]],
    query_item: dict[str, Any],
    k: int,
) -> float:

    return float(
        any(
            result_is_relevant(
                result,
                query_item,
            )
            for result in results[:k]
        )
    )


def reciprocal_rank(
    results: list[dict[str, Any]],
    query_item: dict[str, Any],
) -> float:

    for rank, result in enumerate(
        results,
        start=1,
    ):

        if result_is_relevant(
            result,
            query_item,
        ):

            return 1.0 / rank

    return 0.0


def calculate_metrics(
    all_results: list[
        list[dict[str, Any]]
    ],
    queries: list[
        dict[str, Any]
    ],
) -> dict[str, float]:

    if len(all_results) != len(
        queries
    ):

        raise ValueError(
            "Results/query count mismatch."
        )

    if not queries:

        raise ValueError(
            "No evaluation queries."
        )

    hit1 = []
    hit3 = []
    hit5 = []
    hit10 = []
    mrr = []

    for results, query_item in zip(
        all_results,
        queries,
    ):

        hit1.append(
            hit_at_k(
                results,
                query_item,
                1,
            )
        )

        hit3.append(
            hit_at_k(
                results,
                query_item,
                3,
            )
        )

        hit5.append(
            hit_at_k(
                results,
                query_item,
                5,
            )
        )

        hit10.append(
            hit_at_k(
                results,
                query_item,
                10,
            )
        )

        mrr.append(
            reciprocal_rank(
                results,
                query_item,
            )
        )

    count = len(
        queries
    )

    return {
        "Hit@1": sum(hit1) / count,
        "Hit@3": sum(hit3) / count,
        "Hit@5": sum(hit5) / count,
        "Hit@10": sum(hit10) / count,
        "MRR": sum(mrr) / count,
    }


# ============================================================================
# WEIGHTED RRF
# ============================================================================

def weighted_rrf(
    semantic_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    semantic_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[dict[str, Any]]:

    if semantic_weight < 0:
        raise ValueError(
            "semantic_weight cannot be negative."
        )

    if bm25_weight < 0:
        raise ValueError(
            "bm25_weight cannot be negative."
        )

    if (
        semantic_weight == 0
        and bm25_weight == 0
    ):

        raise ValueError(
            "Both retrieval weights are zero."
        )

    if rrf_k <= 0:
        raise ValueError(
            "rrf_k must be greater than zero."
        )

    scores = {}
    records = {}

    # ---------------------------------------------------------------
    # SEMANTIC
    # ---------------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        result_id = get_result_id(
            result
        )

        scores[result_id] = (
            scores.get(
                result_id,
                0.0,
            )
            + semantic_weight
            / (
                rrf_k + rank
            )
        )

        records.setdefault(
            result_id,
            dict(result),
        )

    # ---------------------------------------------------------------
    # BM25
    # ---------------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1,
    ):

        result_id = get_result_id(
            result
        )

        scores[result_id] = (
            scores.get(
                result_id,
                0.0,
            )
            + bm25_weight
            / (
                rrf_k + rank
            )
        )

        records.setdefault(
            result_id,
            dict(result),
        )

    # ---------------------------------------------------------------
    # SORT
    # ---------------------------------------------------------------

    ranked_ids = sorted(
        scores.keys(),
        key=lambda result_id: (
            -scores[result_id],
            result_id,
        ),
    )

    output = []

    for rank, result_id in enumerate(
        ranked_ids,
        start=1,
    ):

        result = dict(
            records[result_id]
        )

        result[
            "hybrid_score"
        ] = scores[result_id]

        result[
            "hybrid_rank"
        ] = rank

        output.append(
            result
        )

    return output


# ============================================================================
# CONSOLE OUTPUT
# ============================================================================

def source_of(
    result: dict[str, Any],
) -> str:

    metadata = result.get(
        "metadata",
        {},
    )

    return str(
        metadata.get(
            "source",
            "",
        )
    )


def print_top_results(
    label: str,
    results: list[dict[str, Any]],
) -> None:

    print()
    print(f"  {label}")

    for rank, result in enumerate(
        results[
            :DISPLAY_TOP_K
        ],
        start=1,
    ):

        print(
            f"    {rank}. "
            f"{source_of(result)}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 100)
    print(
        "HYBRID RAG - CROSS-ENCODER "
        "RERANKING EVALUATION"
    )
    print("=" * 100)

    # ------------------------------------------------------------------
    # QUERIES
    # ------------------------------------------------------------------

    queries = (
        load_evaluation_queries()
    )

    print()
    print(
        f"[OK] Evaluation queries: "
        f"{len(queries)}"
    )

    print(
        f"[OK] Evaluation file: "
        f"{EVALUATION_PATH}"
    )

    # ------------------------------------------------------------------
    # CORPUS
    # ------------------------------------------------------------------

    print()
    print("Loading corpus...")
    print(
        f"  {CHUNKS_PATH}"
    )

    chunks = load_chunks()

    print(
        f"[OK] Corpus chunks: "
        f"{len(chunks):,}"
    )

    validate_corpus(
        chunks
    )

    # ------------------------------------------------------------------
    # GROUND TRUTH
    # ------------------------------------------------------------------

    validate_ground_truth(
        chunks,
        queries,
    )

    # ------------------------------------------------------------------
    # BM25
    # ------------------------------------------------------------------

    print()
    print(
        "Initializing BM25..."
    )

    bm25 = BM25Retriever(
        chunks
    )

    print(
        "[OK] BM25 ready."
    )

    # ------------------------------------------------------------------
    # SEMANTIC
    # ------------------------------------------------------------------

    print()
    print(
        "Initializing semantic retrieval..."
    )

    semantic = (
        SemanticRetriever()
    )

    print(
        "[OK] Semantic retrieval ready."
    )

    # ------------------------------------------------------------------
    # RERANKER
    # ------------------------------------------------------------------

    print()
    print(
        "Initializing cross-encoder reranker..."
    )

    reranker = (
        CrossEncoderReranker(
            model_name=RERANKER_MODEL,
            max_length=RERANKER_MAX_LENGTH,
            batch_size=RERANKER_BATCH_SIZE,
        )
    )

    print(
        "[OK] Cross-encoder reranker ready."
    )

    # ------------------------------------------------------------------
    # RESULT COLLECTION
    # ------------------------------------------------------------------

    semantic_all = []
    hybrid_all = []
    semantic_reranked_all = []
    hybrid_reranked_all = []

    # ------------------------------------------------------------------
    # EXPERIMENT
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "RUNNING RETRIEVAL + RERANKING EXPERIMENT"
    )
    print("=" * 100)

    for index, query_item in enumerate(
        queries,
        start=1,
    ):

        query = str(
            query_item[
                "query"
            ]
        ).strip()

        query_id = str(
            query_item[
                "id"
            ]
        )

        print()
        print(
            f"[{index:02d}/{len(queries)}] "
            f"{query_id}: {query}"
        )

        # --------------------------------------------------------------
        # SEMANTIC
        # --------------------------------------------------------------

        semantic_results = (
            semantic.search(
                query,
                top_k=SEMANTIC_CANDIDATES,
            )
        )

        if not semantic_results:

            raise RuntimeError(
                f"Semantic retrieval returned "
                f"zero results for {query_id}."
            )

        # --------------------------------------------------------------
        # BM25
        # --------------------------------------------------------------

        bm25_results = (
            bm25.search(
                query,
                top_k=BM25_CANDIDATES,
            )
        )

        if not bm25_results:

            raise RuntimeError(
                f"BM25 returned zero results "
                f"for {query_id}."
            )

        # --------------------------------------------------------------
        # HYBRID
        # --------------------------------------------------------------

        hybrid_results = (
            weighted_rrf(
                semantic_results,
                bm25_results,
                semantic_weight=SEMANTIC_WEIGHT,
                bm25_weight=BM25_WEIGHT,
                rrf_k=RRF_K,
            )
        )

        # --------------------------------------------------------------
        # SEMANTIC + RERANKER
        # --------------------------------------------------------------

        semantic_reranked = (
            reranker.rerank(
                query=query,
                results=semantic_results[
                    :RERANK_CANDIDATES
                ],
                top_k=RERANK_OUTPUT_K,
            )
        )

        # --------------------------------------------------------------
        # HYBRID + RERANKER
        # --------------------------------------------------------------

        hybrid_candidates = (
            hybrid_results[
                :RERANK_CANDIDATES
            ]
        )

        hybrid_reranked = (
            reranker.rerank(
                query=query,
                results=hybrid_candidates,
                top_k=RERANK_OUTPUT_K,
            )
        )

        # --------------------------------------------------------------
        # VALIDATE RESULTS
        # --------------------------------------------------------------

        if not semantic_reranked:

            raise RuntimeError(
                f"Semantic reranker returned "
                f"zero results for {query_id}."
            )

        if not hybrid_reranked:

            raise RuntimeError(
                f"Hybrid reranker returned "
                f"zero results for {query_id}."
            )

        # --------------------------------------------------------------
        # STORE
        # --------------------------------------------------------------

        semantic_all.append(
            semantic_results
        )

        hybrid_all.append(
            hybrid_results
        )

        semantic_reranked_all.append(
            semantic_reranked
        )

        hybrid_reranked_all.append(
            hybrid_reranked
        )

        # --------------------------------------------------------------
        # DISPLAY
        # --------------------------------------------------------------

        print_top_results(
            "Semantic top results:",
            semantic_results,
        )

        print_top_results(
            "Semantic + Reranker:",
            semantic_reranked,
        )

        print_top_results(
            "Hybrid 75/25:",
            hybrid_results,
        )

        print_top_results(
            "Hybrid + Reranker:",
            hybrid_reranked,
        )

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    semantic_metrics = (
        calculate_metrics(
            semantic_all,
            queries,
        )
    )

    hybrid_metrics = (
        calculate_metrics(
            hybrid_all,
            queries,
        )
    )

    semantic_reranked_metrics = (
        calculate_metrics(
            semantic_reranked_all,
            queries,
        )
    )

    hybrid_reranked_metrics = (
        calculate_metrics(
            hybrid_reranked_all,
            queries,
        )
    )

    # ------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "RERANKING RESULTS"
    )
    print("=" * 100)

    print(
        f"{'Method':<28}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
    )

    print("-" * 78)

    methods = [
        (
            "Semantic",
            semantic_metrics,
        ),
        (
            "Hybrid 75/25",
            hybrid_metrics,
        ),
        (
            "Semantic + Reranker",
            semantic_reranked_metrics,
        ),
        (
            "Hybrid + Reranker",
            hybrid_reranked_metrics,
        ),
    ]

    for name, metrics in methods:

        print(
            f"{name:<28}"
            f"{metrics['Hit@1']:>10.3f}"
            f"{metrics['Hit@3']:>10.3f}"
            f"{metrics['Hit@5']:>10.3f}"
            f"{metrics['Hit@10']:>10.3f}"
            f"{metrics['MRR']:>10.3f}"
        )

    # ------------------------------------------------------------------
    # IMPROVEMENT
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "RERANKER IMPROVEMENT"
    )
    print("=" * 100)

    semantic_hit1_delta = (
        semantic_reranked_metrics[
            "Hit@1"
        ]
        - semantic_metrics[
            "Hit@1"
        ]
    )

    semantic_mrr_delta = (
        semantic_reranked_metrics[
            "MRR"
        ]
        - semantic_metrics[
            "MRR"
        ]
    )

    hybrid_hit1_delta = (
        hybrid_reranked_metrics[
            "Hit@1"
        ]
        - hybrid_metrics[
            "Hit@1"
        ]
    )

    hybrid_mrr_delta = (
        hybrid_reranked_metrics[
            "MRR"
        ]
        - hybrid_metrics[
            "MRR"
        ]
    )

    print(
        "Semantic → Semantic + Reranker"
    )

    print(
        f"  Hit@1 change : "
        f"{semantic_hit1_delta:+.3f}"
    )

    print(
        f"  MRR change   : "
        f"{semantic_mrr_delta:+.3f}"
    )

    print()
    print(
        "Hybrid → Hybrid + Reranker"
    )

    print(
        f"  Hit@1 change : "
        f"{hybrid_hit1_delta:+.3f}"
    )

    print(
        f"  MRR change   : "
        f"{hybrid_mrr_delta:+.3f}"
    )

    # ------------------------------------------------------------------
    # BEST
    # ------------------------------------------------------------------

    metrics_by_name = {
        name: metrics
        for name, metrics in methods
    }

    best_name = max(
        metrics_by_name,
        key=lambda name: (
            metrics_by_name[
                name
            ]["MRR"],
            metrics_by_name[
                name
            ]["Hit@5"],
            metrics_by_name[
                name
            ]["Hit@1"],
        ),
    )

    best = metrics_by_name[
        best_name
    ]

    print()
    print("=" * 100)
    print(
        "BEST RETRIEVAL PIPELINE"
    )
    print("=" * 100)

    print(
        f"Method : {best_name}"
    )

    print(
        f"Hit@1  : "
        f"{best['Hit@1']:.3f}"
    )

    print(
        f"Hit@3  : "
        f"{best['Hit@3']:.3f}"
    )

    print(
        f"Hit@5  : "
        f"{best['Hit@5']:.3f}"
    )

    print(
        f"Hit@10 : "
        f"{best['Hit@10']:.3f}"
    )

    print(
        f"MRR    : "
        f"{best['MRR']:.3f}"
    )

    # ------------------------------------------------------------------
    # CONFIG
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "EXPERIMENT CONFIGURATION"
    )
    print("=" * 100)

    print(
        f"Semantic candidates : "
        f"{SEMANTIC_CANDIDATES}"
    )

    print(
        f"BM25 candidates     : "
        f"{BM25_CANDIDATES}"
    )

    print(
        f"Semantic weight     : "
        f"{SEMANTIC_WEIGHT:.2f}"
    )

    print(
        f"BM25 weight         : "
        f"{BM25_WEIGHT:.2f}"
    )

    print(
        f"RRF K               : "
        f"{RRF_K}"
    )

    print(
        f"Rerank candidates   : "
        f"{RERANK_CANDIDATES}"
    )

    print(
        f"Rerank output       : "
        f"{RERANK_OUTPUT_K}"
    )

    print(
        f"Reranker model      : "
        f"{RERANKER_MODEL}"
    )

    # ------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "EXPERIMENT COMPLETE"
    )
    print("=" * 100)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()