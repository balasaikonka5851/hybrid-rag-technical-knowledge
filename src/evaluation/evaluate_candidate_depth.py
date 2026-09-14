"""
HYBRID RAG - CANDIDATE DEPTH EXPERIMENT

Measures whether increasing the number of candidates retrieved by BM25 and
semantic search improves the chance that a ground-truth chunk is present.

This script does NOT modify the retrieval pipeline or any index.
It only runs an experiment against the existing 40-query benchmark.

Run from the project root:
    python src\evaluation\evaluate_candidate_depth.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "queries.json"
FAISS_PATH = PROJECT_ROOT / "vectorstore" / "faiss.index"
METADATA_PATH = PROJECT_ROOT / "vectorstore" / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Candidate depths to test. Each experiment uses the same depth for both
# retrievers so the comparison is easy to interpret.
DEPTHS = [10, 20, 40, 60]

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_source(source: str) -> str:
    """Normalize source paths so ground truth and corpus metadata compare."""
    source = str(source).replace("\\", "/").strip().lower()

    prefixes = [
        "data/raw/",
        "data/processed/",
        "raw/",
        "processed/",
        "fastapi/",
        "python/",
    ]

    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if source.startswith(prefix):
                source = source[len(prefix):]
                changed = True

    return source.lstrip("/")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_query_text(item: dict[str, Any]) -> str:
    for key in ("query", "question", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"Could not find query text in evaluation item: {item}")


def get_ground_truth_sources(item: dict[str, Any]) -> list[str]:
    """
    Supports the common benchmark format:
        "ground_truth": [
            {"source": "..."},
            ...
        ]

    Also accepts strings and a few equivalent field names to make the
    evaluator robust without changing the benchmark.
    """
    gt = item.get("ground_truth", item.get("relevant_sources", []))

    if isinstance(gt, str):
        return [gt]

    if not isinstance(gt, list):
        return []

    sources: list[str] = []

    for entry in gt:
        if isinstance(entry, str):
            sources.append(entry)
            continue

        if isinstance(entry, dict):
            for key in ("source", "file", "path", "document", "doc"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    sources.append(value.strip())
                    break

    return sources


def get_chunk_source(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("source", "file_name", "path"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    for key in ("source", "file_name", "path"):
        value = chunk.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_chunk_technology(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get("technology")
        if isinstance(value, str):
            return value.lower().strip()

    value = chunk.get("technology", "")
    return str(value).lower().strip()


def get_chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk_text"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def source_matches(
    chunk: dict[str, Any],
    ground_truth_sources: set[str],
    technology: str,
) -> bool:
    source = normalize_source(get_chunk_source(chunk))
    if not source:
        return False

    # Ground truth is technology-specific, so prefer matching technology
    # when that metadata exists.
    chunk_tech = get_chunk_technology(chunk)
    if technology and chunk_tech and chunk_tech != technology:
        return False

    return source in ground_truth_sources


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def validate_ground_truth(
    queries: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """Fail early if any benchmark source does not exist in the corpus."""
    available: set[tuple[str, str]] = set()

    for chunk in chunks:
        tech = get_chunk_technology(chunk)
        source = normalize_source(get_chunk_source(chunk))
        if source:
            available.add((tech, source))

    missing: list[str] = []

    for item in queries:
        qid = str(item.get("id", "?"))
        tech = str(item.get("technology", "")).lower().strip()

        for source in get_ground_truth_sources(item):
            normalized = normalize_source(source)
            if (tech, normalized) not in available:
                missing.append(f"{qid}: {source}")

    if missing:
        print("[ERROR] Ground-truth validation failed.")
        for entry in missing:
            print(f"  - {entry}")
        raise RuntimeError(f"{len(missing)} ground-truth source declarations are missing.")

    print("[OK] Ground-truth validated against corpus.")


def reciprocal_rank_fusion(
    semantic_results: list[int],
    bm25_results: list[int],
    semantic_weight: float = 0.75,
    bm25_weight: float = 0.25,
    rrf_k: int = 60,
) -> list[int]:
    """Fuse candidate rankings using weighted RRF."""
    scores: dict[int, float] = {}

    for rank, idx in enumerate(semantic_results, start=1):
        scores[idx] = scores.get(idx, 0.0) + semantic_weight / (rrf_k + rank)

    for rank, idx in enumerate(bm25_results, start=1):
        scores[idx] = scores.get(idx, 0.0) + bm25_weight / (rrf_k + rank)

    return [
        idx
        for idx, _ in sorted(
            scores.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def compute_metrics(
    ranked_indices: list[int],
    query: dict[str, Any],
    chunks: list[dict[str, Any]],
) -> tuple[int, int, int, int, float]:
    """Return Hit@1, Hit@3, Hit@5, Hit@10 and reciprocal rank."""
    technology = str(query.get("technology", "")).lower().strip()
    gt_sources = {
        normalize_source(source)
        for source in get_ground_truth_sources(query)
    }

    relevant = [
        source_matches(chunks[idx], gt_sources, technology)
        for idx in ranked_indices
    ]

    def hit(k: int) -> int:
        return int(any(relevant[:k]))

    first_rank = None
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            first_rank = rank
            break

    rr = 0.0 if first_rank is None else 1.0 / first_rank

    return hit(1), hit(3), hit(5), hit(10), rr


def aggregate(rows: list[tuple[int, int, int, int, float]]) -> tuple[float, ...]:
    if not rows:
        return (0.0,) * 5

    arr = np.asarray(rows, dtype=np.float64)
    return tuple(arr.mean(axis=0).tolist())


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 100)
    print("HYBRID RAG - CANDIDATE DEPTH EXPERIMENT")
    print("=" * 100)
    print()

    print(f"[OK] Project root: {PROJECT_ROOT}")
    print(f"[OK] Evaluation file: {QUERIES_PATH}")
    print()

    queries_raw = load_json(QUERIES_PATH)
    chunks_raw = load_json(CHUNKS_PATH)

    if not isinstance(queries_raw, list):
        raise ValueError("queries.json must contain a JSON list.")

    if not isinstance(chunks_raw, list):
        raise ValueError("chunks.json must contain a JSON list.")

    queries = queries_raw
    chunks = chunks_raw

    print(f"[OK] Evaluation queries: {len(queries)}")
    print(f"[OK] Corpus chunks: {len(chunks)}")

    validate_ground_truth(queries, chunks)
    print()

    # -----------------------------------------------------------------------
    # BM25
    # -----------------------------------------------------------------------
    print("Initializing BM25...")
    tokenized = [tokenize(get_chunk_text(chunk)) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    print("[OK] BM25 ready.")
    print()

    # -----------------------------------------------------------------------
    # FAISS + embedding model
    # -----------------------------------------------------------------------
    print("Loading FAISS...")
    if not FAISS_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {FAISS_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    index = faiss.read_index(str(FAISS_PATH))
    metadata = load_json(METADATA_PATH)

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS/corpus mismatch: FAISS={index.ntotal}, chunks={len(chunks)}"
        )

    if len(metadata) != len(chunks):
        raise RuntimeError(
            f"Metadata/corpus mismatch: metadata={len(metadata)}, chunks={len(chunks)}"
        )

    print(f"[OK] FAISS vectors: {index.ntotal}")
    print("[OK] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"[OK] Embedding model: {EMBEDDING_MODEL}")
    print()

    # -----------------------------------------------------------------------
    # Collect all candidate rankings once.
    #
    # We retrieve the maximum depth once and then slice it for 10/20/40/60.
    # This guarantees that all depth comparisons use identical underlying
    # rankings and avoids repeatedly encoding the same queries.
    # -----------------------------------------------------------------------
    max_depth = max(DEPTHS)

    print("=" * 100)
    print("COLLECTING TOP CANDIDATES")
    print("=" * 100)

    all_candidates: dict[str, dict[str, list[int]]] = {}

    for position, query in enumerate(queries, start=1):
        qid = str(query.get("id", position))
        query_text = get_query_text(query)

        # BM25
        bm25_scores = bm25.get_scores(tokenize(query_text))
        bm25_top = np.argsort(-np.asarray(bm25_scores))[:max_depth]
        bm25_indices = [int(i) for i in bm25_top]

        # Semantic
        embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        semantic_scores, semantic_indices = index.search(embedding, max_depth)

        # FAISS returns shape (1, K).
        semantic_indices = semantic_indices[0]
        semantic_indices = [
            int(i) for i in semantic_indices if int(i) >= 0
        ]

        all_candidates[qid] = {
            "semantic": semantic_indices,
            "bm25": bm25_indices,
        }

        print(f"[{position:02d}/{len(queries)}] {qid}: {query_text}")

    print()
    print("[OK] Candidate collection completed.")
    print()

    # -----------------------------------------------------------------------
    # Evaluate individual retrievers and weighted hybrid at each depth.
    # -----------------------------------------------------------------------
    print("=" * 100)
    print("DEPTH EXPERIMENT")
    print("=" * 100)
    print()

    print(
        f"{'Depth':>7} | "
        f"{'Semantic H@1':>12} {'Semantic H@5':>12} {'Semantic H@10':>13} {'Semantic MRR':>12} | "
        f"{'Hybrid H@1':>10} {'Hybrid H@5':>10} {'Hybrid H@10':>11} {'Hybrid MRR':>10}"
    )
    print("-" * 120)

    summary: list[dict[str, Any]] = []

    for depth in DEPTHS:
        semantic_rows = []
        hybrid_rows = []

        for query in queries:
            qid = str(query.get("id"))
            candidates = all_candidates[qid]

            semantic_ranked = candidates["semantic"][:depth]
            bm25_ranked = candidates["bm25"][:depth]

            semantic_metrics = compute_metrics(
                semantic_ranked, query, chunks
            )

            hybrid_ranked = reciprocal_rank_fusion(
                semantic_ranked,
                bm25_ranked,
                semantic_weight=0.75,
                bm25_weight=0.25,
                rrf_k=60,
            )

            hybrid_metrics = compute_metrics(
                hybrid_ranked, query, chunks
            )

            semantic_rows.append(semantic_metrics)
            hybrid_rows.append(hybrid_metrics)

        sh1, sh3, sh5, sh10, smrr = aggregate(semantic_rows)
        hh1, hh3, hh5, hh10, hmrr = aggregate(hybrid_rows)

        print(
            f"{depth:7d} | "
            f"{sh1:12.3f} {sh5:12.3f} {sh10:13.3f} {smrr:12.3f} | "
            f"{hh1:10.3f} {hh5:10.3f} {hh10:11.3f} {hmrr:10.3f}"
        )

        summary.append(
            {
                "depth": depth,
                "semantic": {
                    "hit_at_1": sh1,
                    "hit_at_3": sh3,
                    "hit_at_5": sh5,
                    "hit_at_10": sh10,
                    "mrr": smrr,
                },
                "hybrid_75_25": {
                    "hit_at_1": hh1,
                    "hit_at_3": hh3,
                    "hit_at_5": hh5,
                    "hit_at_10": hh10,
                    "mrr": hmrr,
                },
            }
        )

    print()
    print("=" * 100)
    print("INTERPRETATION")
    print("=" * 100)

    best_hybrid = max(
        summary,
        key=lambda row: (
            row["hybrid_75_25"]["mrr"],
            row["hybrid_75_25"]["hit_at_10"],
        ),
    )

    best_semantic = max(
        summary,
        key=lambda row: (
            row["semantic"]["mrr"],
            row["semantic"]["hit_at_10"],
        ),
    )

    print(
        f"[BEST HYBRID] depth={best_hybrid['depth']} | "
        f"MRR={best_hybrid['hybrid_75_25']['mrr']:.3f} | "
        f"Hit@10={best_hybrid['hybrid_75_25']['hit_at_10']:.3f}"
    )

    print(
        f"[BEST SEMANTIC] depth={best_semantic['depth']} | "
        f"MRR={best_semantic['semantic']['mrr']:.3f} | "
        f"Hit@10={best_semantic['semantic']['hit_at_10']:.3f}"
    )

    print()
    print("If Hit@10 rises substantially with depth while Hit@1/MRR stay")
    print("similar, the main problem is likely ranking rather than candidate recall.")
    print("If Hit@10 itself remains low, the retrievers are failing to retrieve")
    print("the relevant source and we should improve retrieval/query handling.")
    print()
    print("[OK] Experiment complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Experiment interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)
