"""
HybridRAG - FAISS Vector Index Builder

Builds a persistent FAISS index from normalized embeddings.

Input:
    vectorstore/embeddings.npy
    vectorstore/metadata.json

Output:
    vectorstore/faiss.index
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

EMBEDDINGS_FILE = VECTORSTORE_DIR / "embeddings.npy"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"
FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss.index"


# ============================================================
# CONFIGURATION
# ============================================================

EXPECTED_DIMENSION = 384

INDEX_TYPE = "IndexFlatIP"


# ============================================================
# LOGGING
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings() -> np.ndarray:
    """
    Load and validate embeddings.npy.
    """

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file not found:\n"
            f"{EMBEDDINGS_FILE}\n\n"
            "Run the embedding pipeline first."
        )

    print(f"Loading embeddings:")
    print(f"  {EMBEDDINGS_FILE}")

    try:
        embeddings = np.load(
            EMBEDDINGS_FILE,
            allow_pickle=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embeddings.npy:\n{exc}"
        ) from exc

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a 2-dimensional matrix.\n"
            f"Received shape: {embeddings.shape}"
        )

    vector_count, dimension = embeddings.shape

    if dimension != EXPECTED_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension.\n"
            f"Expected: {EXPECTED_DIMENSION}\n"
            f"Received: {dimension}"
        )

    if vector_count == 0:
        raise ValueError(
            "Embeddings file contains zero vectors."
        )

    # --------------------------------------------------------
    # Numerical validation
    # --------------------------------------------------------

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    # --------------------------------------------------------
    # Normalization validation
    # --------------------------------------------------------

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-3,
    ):
        raise ValueError(
            "Embeddings are not normalized.\n"
            "The FAISS Inner Product index expects normalized vectors."
        )

    print(f"[OK] Loaded {vector_count:,} vectors.")
    print(f"[OK] Dimension: {dimension}")
    print("[OK] dtype: float32")
    print("[OK] All values are finite.")
    print("[OK] Vectors are normalized.")

    return embeddings


# ============================================================
# LOAD METADATA
# ============================================================

def load_metadata(expected_count: int) -> list[dict[str, Any]]:
    """
    Load metadata and verify alignment with embeddings.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found:\n"
            f"{METADATA_FILE}"
        )

    print()
    print(f"Loading metadata:")
    print(f"  {METADATA_FILE}")

    try:
        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:
            metadata = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"metadata.json contains invalid JSON:\n{exc}"
        ) from exc

    if not isinstance(metadata, list):
        raise ValueError(
            "metadata.json must contain a JSON list."
        )

    if len(metadata) != expected_count:
        raise ValueError(
            "Embedding/metadata count mismatch.\n"
            f"Embeddings: {expected_count:,}\n"
            f"Metadata:   {len(metadata):,}"
        )

    # --------------------------------------------------------
    # Verify vector indexes
    # --------------------------------------------------------

    for expected_index, item in enumerate(metadata):

        if not isinstance(item, dict):
            raise ValueError(
                f"Metadata item {expected_index} is invalid."
            )

        actual_index = item.get("vector_index")

        if actual_index != expected_index:
            raise ValueError(
                "Vector/metadata alignment error.\n"
                f"Expected vector index: {expected_index}\n"
                f"Received: {actual_index}"
            )

        if not item.get("chunk_id"):
            raise ValueError(
                f"Metadata item {expected_index} "
                "has no chunk_id."
            )

        if not isinstance(item.get("text"), str):
            raise ValueError(
                f"Metadata item {expected_index} "
                "has invalid text."
            )

    print(f"[OK] Loaded {len(metadata):,} metadata records.")
    print("[OK] Vector/metadata alignment verified.")

    return metadata


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(
    embeddings: np.ndarray,
) -> faiss.Index:
    """
    Build an exact Inner Product FAISS index.

    Because embeddings are normalized, Inner Product gives
    cosine similarity.
    """

    dimension = embeddings.shape[1]

    print()
    print("Building FAISS index...")
    print(f"  Index type : {INDEX_TYPE}")
    print(f"  Dimension  : {dimension}")
    print(f"  Vectors    : {len(embeddings):,}")

    index = faiss.IndexFlatIP(
        dimension
    )

    if index.is_trained is not True:
        raise RuntimeError(
            "FAISS index is unexpectedly not trained."
        )

    index.add(embeddings)

    if index.ntotal != len(embeddings):
        raise ValueError(
            "FAISS vector count mismatch.\n"
            f"Expected: {len(embeddings):,}\n"
            f"FAISS:    {index.ntotal:,}"
        )

    print("[OK] FAISS index built.")
    print(f"[OK] Total indexed vectors: {index.ntotal:,}")

    return index


# ============================================================
# ATOMIC INDEX SAVE
# ============================================================

def save_index(
    index: faiss.Index,
) -> None:
    """
    Save FAISS index atomically.
    """

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_fd, temp_name = tempfile.mkstemp(
        suffix=".index",
        dir=str(VECTORSTORE_DIR),
    )

    os.close(temp_fd)

    temp_path = Path(temp_name)

    try:

        faiss.write_index(
            index,
            str(temp_path),
        )

        os.replace(
            temp_path,
            FAISS_INDEX_FILE,
        )

    finally:

        if temp_path.exists():
            temp_path.unlink()

    print()
    print("[OK] FAISS index saved:")
    print(f"     {FAISS_INDEX_FILE}")


# ============================================================
# VERIFY SAVED INDEX
# ============================================================

def verify_index(
    expected_count: int,
    expected_dimension: int,
) -> None:
    """
    Reload the index from disk and verify it.
    """

    print()
    print("Verifying saved FAISS index...")

    if not FAISS_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index was not created:\n"
            f"{FAISS_INDEX_FILE}"
        )

    try:
        index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to reload FAISS index:\n{exc}"
        ) from exc

    if index.ntotal != expected_count:
        raise ValueError(
            "Saved FAISS index contains the wrong "
            "number of vectors.\n"
            f"Expected: {expected_count:,}\n"
            f"Received: {index.ntotal:,}"
        )

    if index.d != expected_dimension:
        raise ValueError(
            "Saved FAISS index has the wrong dimension.\n"
            f"Expected: {expected_dimension}\n"
            f"Received: {index.d}"
        )

    if not index.is_trained:
        raise ValueError(
            "Saved FAISS index is not trained."
        )

    print("[OK] Index successfully reloaded.")
    print(f"[OK] Vector count: {index.ntotal:,}")
    print(f"[OK] Dimension: {index.d}")
    print("[OK] Index is trained.")


# ============================================================
# TEST SEARCH
# ============================================================

def test_search(
    index: faiss.Index,
    metadata: list[dict[str, Any]],
    k: int = 5,
) -> None:
    """
    Perform a small sanity-check search using one existing
    embedding.

    This confirms that the index can actually retrieve vectors
    and map them back to metadata.
    """

    print()
    print("Running FAISS sanity search...")

    if index.ntotal == 0:
        raise ValueError(
            "Cannot test an empty FAISS index."
        )

    k = min(
        k,
        index.ntotal,
    )

    # Reconstruct the first vector from the index.
    query_vector = np.asarray(
        index.reconstruct(0),
        dtype=np.float32,
    ).reshape(1, -1)

    scores, indices = index.search(
        query_vector,
        k,
    )

    print()
    print(f"Top {k} results:")
    print("-" * 70)

    for rank, (score, vector_index) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):

        if vector_index < 0:
            continue

        item = metadata[vector_index]

        technology = item["metadata"]["technology"]
        source = item["metadata"]["source"]
        chunk_id = item["chunk_id"]

        print(
            f"{rank}. score={score:.4f} | "
            f"vector={vector_index} | "
            f"{technology} | "
            f"{source}"
        )

        print(
            f"   chunk_id={chunk_id}"
        )

    # The first vector searching for itself should have
    # similarity extremely close to 1.
    first_score = float(scores[0][0])

    if first_score < 0.99:
        raise ValueError(
            "FAISS sanity search failed.\n"
            f"Expected self-similarity near 1.0, "
            f"received {first_score:.4f}"
        )

    print()
    print(
        f"[OK] Self-similarity check passed: "
        f"{first_score:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print_header(
        "HYBRID RAG - FAISS INDEX BUILDER"
    )

    try:

        # ----------------------------------------------------
        # 1. Load embeddings
        # ----------------------------------------------------

        print("[1/5] Loading embeddings...")

        embeddings = load_embeddings()

        # ----------------------------------------------------
        # 2. Load metadata
        # ----------------------------------------------------

        print()
        print("[2/5] Loading metadata...")

        metadata = load_metadata(
            expected_count=len(embeddings)
        )

        # ----------------------------------------------------
        # 3. Build index
        # ----------------------------------------------------

        print()
        print("[3/5] Building FAISS index...")

        index = build_index(
            embeddings
        )

        # ----------------------------------------------------
        # 4. Save and verify
        # ----------------------------------------------------

        print()
        print("[4/5] Saving FAISS index...")

        save_index(
            index
        )

        verify_index(
            expected_count=len(embeddings),
            expected_dimension=embeddings.shape[1],
        )

        # ----------------------------------------------------
        # 5. Sanity search
        # ----------------------------------------------------

        print()
        print("[5/5] Testing FAISS retrieval...")

        test_search(
            index=index,
            metadata=metadata,
            k=5,
        )

        print()
        print_header(
            "FAISS INDEX BUILD COMPLETED SUCCESSFULLY"
        )

        print(
            f"Vectors indexed : {index.ntotal:,}"
        )

        print(
            f"Dimension        : {index.d}"
        )

        print(
            f"Index type       : {INDEX_TYPE}"
        )

        print(
            f"Index file       : {FAISS_INDEX_FILE}"
        )

        print()
        print(
            "[OK] Semantic retrieval foundation is ready."
        )

    except KeyboardInterrupt:

        print()
        print(
            "[STOPPED] FAISS process interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:

        print()
        print_header(
            "FAISS INDEX BUILD FAILED"
        )

        print(f"[ERROR] {exc}")
        print()
        print(
            "The FAISS index was not confirmed as valid."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()