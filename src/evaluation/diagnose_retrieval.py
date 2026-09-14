"""
HYBRID RAG - RETRIEVAL DIAGNOSTICS

Purpose:
    Diagnose WHY relevant documents are not appearing near the top of
    semantic, BM25, or weighted-RRF rankings.

For each benchmark query, this script reports:
    - ground-truth source(s)
    - semantic rank of the first ground-truth source
    - BM25 rank of the first ground-truth source
    - hybrid RRF rank of the first ground-truth source
    - whether the ground truth was retrieved within top-60
    - top-ranked source for each method

It does not modify indexes, chunks, or evaluation data.

Run from project root:
    python src\evaluation\diagnose_retrieval.py
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.json"
QUERIES_PATH = PROJECT_ROOT / "data" / "evaluation" / "queries.json"
FAISS_PATH = PROJECT_ROOT / "vectorstore" / "faiss.index"
METADATA_PATH = PROJECT_ROOT / "vectorstore" / "metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEPTH = 60
SEMANTIC_WEIGHT = 0.75
BM25_WEIGHT = 0.25
RRF_K = 60

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def normalize_source(source: str) -> str:
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


def query_text(item: dict[str, Any]) -> str:
    for key in ("query", "question", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No query text found: {item}")


def ground_truth_sources(item: dict[str, Any]) -> list[str]:
    gt = item.get("ground_truth", item.get("relevant_sources", []))

    if isinstance(gt, str):
        return [gt]

    if not isinstance(gt, list):
        return []

    sources = []

    for entry in gt:
        if isinstance(entry, str):
            sources.append(entry)
        elif isinstance(entry, dict):
            for key in ("source", "file", "path", "document", "doc"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    sources.append(value.strip())
                    break

    return sources


def chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("text", "content", "chunk_text"):
        value = chunk.get(key)
        if isinstance(value, str):
            return value
    return ""


def chunk_source(chunk: dict[str, Any]) -> str:
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


def chunk_technology(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})

    if isinstance(metadata, dict):
        value = metadata.get("technology")
        if isinstance(value, str):
            return value.lower().strip()

    return str(chunk.get("technology", "")).lower().strip()


def is_relevant(
    chunk: dict[str, Any],
    gt_sources: set[str],
    technology: str,
) -> bool:
    if technology:
        chunk_tech = chunk_technology(chunk)
        if chunk_tech and chunk_tech != technology:
            return False

    return normalize_source(chunk_source(chunk)) in gt_sources


def weighted_rrf(
    semantic_indices: list[int],
    bm25_indices: list[int],
) -> list[int]:
    scores: dict[int, float] = {}

    for rank, idx in enumerate(semantic_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + (
            SEMANTIC_WEIGHT / (RRF_K + rank)
        )

    for rank, idx in enumerate(bm25_indices, start=1):
        scores[idx] = scores.get(idx, 0.0) + (
            BM25_WEIGHT / (RRF_K + rank)
        )

    return [
        idx
        for idx, _ in sorted(
            scores.items(),
            key=lambda pair: (-pair[1], pair[0]),
        )
    ]


def first_relevant_rank(
    ranking: list[int],
    chunks: list[dict[str, Any]],
    gt_sources: set[str],
    technology: str,
) -> int | None:
    for rank, idx in enumerate(ranking, start=1):
        if is_relevant(chunks[idx], gt_sources, technology):
            return rank
    return None


def source_label(chunk: dict[str, Any]) -> str:
    source = normalize_source(chunk_source(chunk))
    if source:
        return source
    return "<unknown-source>"


def preview(text: str, limit: int = 105) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def main() -> None:
    print("=" * 100)
    print("HYBRID RAG - RETRIEVAL DIAGNOSTICS")
    print("=" * 100)
    print()

    queries = load_json(QUERIES_PATH)
    chunks = load_json(CHUNKS_PATH)

    if not isinstance(queries, list):
        raise ValueError("queries.json must contain a JSON list.")

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a JSON list.")

    print(f"[OK] Evaluation queries: {len(queries)}")
    print(f"[OK] Corpus chunks: {len(chunks)}")
    print(f"[OK] Diagnostic depth: top-{DEPTH}")
    print(
        f"[OK] Hybrid weights: semantic={SEMANTIC_WEIGHT:.2f}, "
        f"BM25={BM25_WEIGHT:.2f}, RRF-k={RRF_K}"
    )
    print()

    # Validate all GT sources before doing retrieval.
    available = set()

    for chunk in chunks:
        available.add(
            (
                chunk_technology(chunk),
                normalize_source(chunk_source(chunk)),
            )
        )

    missing = []

    for item in queries:
        qid = str(item.get("id", "?"))
        technology = str(item.get("technology", "")).lower().strip()

        for source in ground_truth_sources(item):
            normalized = normalize_source(source)
            if (technology, normalized) not in available:
                missing.append(f"{qid}: {source}")

    if missing:
        print("[ERROR] Ground-truth validation failed:")
        for entry in missing:
            print(f"  - {entry}")
        raise RuntimeError("Ground-truth contains missing sources.")

    print("[OK] Ground-truth validated.")
    print()

    # BM25
    print("Initializing BM25...")
    bm25 = BM25Okapi([tokenize(chunk_text(c)) for c in chunks])
    print("[OK] BM25 ready.")
    print()

    # FAISS
    print("Loading FAISS...")
    index = faiss.read_index(str(FAISS_PATH))

    if index.ntotal != len(chunks):
        raise RuntimeError(
            f"FAISS/corpus mismatch: {index.ntotal} vs {len(chunks)}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata: {METADATA_PATH}")

    metadata = load_json(METADATA_PATH)

    if len(metadata) != len(chunks):
        raise RuntimeError(
            f"Metadata/corpus mismatch: {len(metadata)} vs {len(chunks)}"
        )

    print(f"[OK] FAISS vectors: {index.ntotal}")
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"[OK] Model: {MODEL_NAME}")
    print()

    print("=" * 100)
    print("QUERY-BY-QUERY DIAGNOSTICS")
    print("=" * 100)

    records = []

    for number, item in enumerate(queries, start=1):
        qid = str(item.get("id", number))
        technology = str(item.get("technology", "")).lower().strip()
        text = query_text(item)

        gt_sources = {
            normalize_source(source)
            for source in ground_truth_sources(item)
        }

        # BM25
        scores = bm25.get_scores(tokenize(text))
        bm25_indices = [
            int(i)
            for i in np.argsort(-np.asarray(scores))[:DEPTH]
        ]

        # Semantic
        embedding = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        semantic_scores, semantic_indices = index.search(
            embedding, DEPTH
        )

        semantic_indices = [
            int(i)
            for i in semantic_indices[0]
            if int(i) >= 0
        ]

        # Hybrid
        hybrid_indices = weighted_rrf(
            semantic_indices,
            bm25_indices,
        )

        semantic_rank = first_relevant_rank(
            semantic_indices,
            chunks,
            gt_sources,
            technology,
        )

        bm25_rank = first_relevant_rank(
            bm25_indices,
            chunks,
            gt_sources,
            technology,
        )

        hybrid_rank = first_relevant_rank(
            hybrid_indices,
            chunks,
            gt_sources,
            technology,
        )

        semantic_top = chunks[semantic_indices[0]]
        bm25_top = chunks[bm25_indices[0]]
        hybrid_top = chunks[hybrid_indices[0]]

        records.append(
            {
                "qid": qid,
                "query": text,
                "semantic_rank": semantic_rank,
                "bm25_rank": bm25_rank,
                "hybrid_rank": hybrid_rank,
                "semantic_top": source_label(semantic_top),
                "bm25_top": source_label(bm25_top),
                "hybrid_top": source_label(hybrid_top),
                "gt_sources": sorted(gt_sources),
            }
        )

        def fmt_rank(rank: int | None) -> str:
            return "-" if rank is None else str(rank)

        print()
        print(f"[{number:02d}/{len(queries)}] {qid}")
        print(f"Query: {text}")
        print(
            "GT: "
            + ", ".join(sorted(gt_sources))
        )
        print(
            f"  Semantic: rank={fmt_rank(semantic_rank):>2} | "
            f"top={semantic_top and source_label(semantic_top)}"
        )
        print(
            f"  BM25:     rank={fmt_rank(bm25_rank):>2} | "
            f"top={source_label(bm25_top)}"
        )
        print(
            f"  Hybrid:   rank={fmt_rank(hybrid_rank):>2} | "
            f"top={source_label(hybrid_top)}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 100)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 100)

    methods = [
        ("Semantic", "semantic_rank"),
        ("BM25", "bm25_rank"),
        ("Hybrid", "hybrid_rank"),
    ]

    for name, key in methods:
        ranks = [
            record[key]
            for record in records
            if record[key] is not None
        ]

        top10 = sum(rank <= 10 for rank in ranks)
        top20 = sum(rank <= 20 for rank in ranks)
        top40 = sum(rank <= 40 for rank in ranks)
        top60 = sum(rank <= 60 for rank in ranks)

        print(
            f"{name:<10} | "
            f"found={len(ranks):2d}/{len(records)} | "
            f"top10={top10:2d} | "
            f"top20={top20:2d} | "
            f"top40={top40:2d} | "
            f"top60={top60:2d}"
        )

    print()
    print("=" * 100)
    print("RANKING-BOTTLENECK QUERIES")
    print("=" * 100)

    # Queries where Hybrid finds the GT within 60 but not in top 5.
    bottlenecks = [
        record
        for record in records
        if record["hybrid_rank"] is not None
        and record["hybrid_rank"] > 5
    ]

    if bottlenecks:
        print(
            "These queries retrieve a ground-truth source, but Hybrid ranks "
            "it below position 5:"
        )

        for record in bottlenecks:
            print(
                f"  {record['qid']}: "
                f"Hybrid rank={record['hybrid_rank']} | "
                f"Semantic={record['semantic_rank'] or '-'} | "
                f"BM25={record['bm25_rank'] or '-'}"
            )
    else:
        print("[NONE] No ranking bottlenecks below top-5.")

    print()
    print("=" * 100)
    print("RETRIEVAL-FAILURE QUERIES")
    print("=" * 100)

    failures = [
        record
        for record in records
        if record["hybrid_rank"] is None
    ]

    if failures:
        print(
            "These queries did NOT retrieve a ground-truth source "
            "within the top-60 Hybrid candidate pool:"
        )

        for record in failures:
            print(
                f"  {record['qid']}: {record['query']}"
            )
    else:
        print("[NONE] Hybrid retrieved a ground-truth source for every query.")

    print()
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Experiment interrupted by user.")
        sys.exit(130)
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)
