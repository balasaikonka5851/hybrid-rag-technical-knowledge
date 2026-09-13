"""
HybridRAG - Embedding Pipeline

Creates dense vector embeddings for all validated chunks using
Sentence Transformers and saves them for later FAISS indexing.

Input:
    data/processed/chunks.json

Output:
    vectorstore/embeddings.npy
    vectorstore/metadata.json
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "chunks.json"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

EMBEDDINGS_FILE = VECTORSTORE_DIR / "embeddings.npy"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64

EXPECTED_DIMENSION = 384

MIN_TEXT_LENGTH = 1


# ============================================================
# LOGGING
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# DEVICE
# ============================================================

def get_device() -> str:
    """
    Automatically select CUDA when available, otherwise CPU.
    """

    if torch.cuda.is_available():
        device = "cuda"

        print(f"[OK] CUDA available")
        print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

        return device

    print("[INFO] CUDA not available.")
    print("[INFO] Using CPU for embeddings.")

    return "cpu"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[dict[str, Any]]:
    """
    Load and validate chunks.json.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run the chunking pipeline first."
        )

    print(f"Loading chunks from:")
    print(f"  {INPUT_FILE}")

    try:
        with INPUT_FILE.open("r", encoding="utf-8") as file:
            chunks = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"chunks.json contains invalid JSON.\n"
            f"JSON error: {exc}"
        ) from exc

    if not isinstance(chunks, list):
        raise ValueError(
            "Invalid chunks.json format.\n"
            "Expected a JSON list of chunks."
        )

    if not chunks:
        raise ValueError(
            "chunks.json is empty. No chunks available for embedding."
        )

    print(f"[OK] Loaded {len(chunks):,} chunks.")

    return chunks


# ============================================================
# CHUNK VALIDATION
# ============================================================

def validate_chunks(chunks: list[dict[str, Any]]) -> None:
    """
    Validate the exact chunk schema produced by chunker.py.
    """

    required_top_level = {"id", "text", "metadata"}

    required_metadata = {
        "document_id",
        "technology",
        "source",
        "file_name",
        "file_type",
        "section",
        "section_level",
        "chunk_index",
    }

    ids: set[str] = set()

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} is not a dictionary."
            )

        missing = required_top_level - chunk.keys()

        if missing:
            raise ValueError(
                f"Chunk {index} is missing fields: {sorted(missing)}"
            )

        chunk_id = chunk["id"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk {index} has an invalid ID."
            )

        if chunk_id in ids:
            raise ValueError(
                f"Duplicate chunk ID detected: {chunk_id}"
            )

        ids.add(chunk_id)

        text = chunk["text"]

        if not isinstance(text, str):
            raise ValueError(
                f"Chunk {index} has non-string text."
            )

        if len(text.strip()) < MIN_TEXT_LENGTH:
            raise ValueError(
                f"Chunk {index} contains empty text."
            )

        metadata = chunk["metadata"]

        if not isinstance(metadata, dict):
            raise ValueError(
                f"Chunk {index} metadata must be a dictionary."
            )

        missing_metadata = required_metadata - metadata.keys()

        if missing_metadata:
            raise ValueError(
                f"Chunk {index} is missing metadata fields: "
                f"{sorted(missing_metadata)}"
            )

        technology = metadata["technology"]

        if technology not in {"python", "fastapi"}:
            raise ValueError(
                f"Chunk {index} has unexpected technology: "
                f"{technology!r}"
            )

    print("[OK] Chunk schema validated.")
    print(f"[OK] Unique chunk IDs: {len(ids):,}")


# ============================================================
# MODEL LOADING
# ============================================================

def load_embedding_model(device: str) -> SentenceTransformer:
    """
    Load the Sentence Transformer model.
    """

    print()
    print(f"Loading embedding model:")
    print(f"  {MODEL_NAME}")
    print(f"  Device: {device}")

    try:
        model = SentenceTransformer(
            MODEL_NAME,
            device=device,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the embedding model.\n\n"
            "Possible causes:\n"
            "1. Internet connection unavailable on first download.\n"
            "2. Hugging Face model download failed.\n"
            "3. PyTorch / Transformers compatibility issue.\n\n"
            f"Original error: {exc}"
        ) from exc

    dimension = model.get_sentence_embedding_dimension()

    print(f"[OK] Model loaded.")
    print(f"[OK] Embedding dimension: {dimension}")

    if dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Unexpected embedding dimension.\n"
            f"Expected: {EXPECTED_DIMENSION}\n"
            f"Received: {dimension}"
        )

    return model


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embeddings(
    model: SentenceTransformer,
    chunks: list[dict[str, Any]],
) -> np.ndarray:
    """
    Generate normalized embeddings while preserving chunk order.
    """

    texts = [chunk["text"] for chunk in chunks]

    total = len(texts)

    print()
    print("Generating embeddings...")
    print(f"  Chunks:     {total:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()

    try:
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding generation failed.\n"
            f"Original error: {exc}"
        ) from exc

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    print()
    print("[OK] Embedding generation completed.")

    # --------------------------------------------------------
    # Shape validation
    # --------------------------------------------------------

    expected_shape = (len(chunks), EXPECTED_DIMENSION)

    if embeddings.shape != expected_shape:
        raise ValueError(
            "Embedding shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {embeddings.shape}"
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
            "Embeddings are not properly normalized."
        )

    print(f"[OK] Shape: {embeddings.shape}")
    print("[OK] Data type: float32")
    print("[OK] All values are finite.")
    print("[OK] Embeddings are normalized.")

    return embeddings


# ============================================================
# METADATA
# ============================================================

def build_metadata(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build metadata aligned exactly with embedding row indexes.

    metadata[i] corresponds to embeddings[i].
    """

    metadata = []

    for vector_index, chunk in enumerate(chunks):

        metadata.append(
            {
                "vector_index": vector_index,
                "chunk_id": chunk["id"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
            }
        )

    if len(metadata) != len(chunks):
        raise ValueError(
            "Metadata count does not match chunk count."
        )

    print(
        f"[OK] Metadata records created: "
        f"{len(metadata):,}"
    )

    return metadata


# ============================================================
# ATOMIC SAVE HELPERS
# ============================================================

def atomic_save_numpy(
    array: np.ndarray,
    destination: Path,
) -> None:
    """
    Save NumPy array atomically.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_fd, temp_name = tempfile.mkstemp(
        suffix=".npy",
        dir=str(destination.parent),
    )

    os.close(temp_fd)

    temp_path = Path(temp_name)

    try:
        np.save(
            temp_path,
            array,
            allow_pickle=False,
        )

        os.replace(
            temp_path,
            destination,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def atomic_save_json(
    data: Any,
    destination: Path,
) -> None:
    """
    Save JSON atomically.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_fd, temp_name = tempfile.mkstemp(
        suffix=".json",
        dir=str(destination.parent),
    )

    os.close(temp_fd)

    temp_path = Path(temp_name)

    try:
        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(
            temp_path,
            destination,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


# ============================================================
# SAVE VECTORSTORE
# ============================================================

def save_vectorstore(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Saving vectorstore...")

    atomic_save_numpy(
        embeddings,
        EMBEDDINGS_FILE,
    )

    atomic_save_json(
        metadata,
        METADATA_FILE,
    )

    print(f"[OK] Embeddings saved:")
    print(f"     {EMBEDDINGS_FILE}")

    print(f"[OK] Metadata saved:")
    print(f"     {METADATA_FILE}")


# ============================================================
# VERIFY SAVED FILES
# ============================================================

def verify_saved_files(
    expected_count: int,
) -> None:

    print()
    print("Verifying saved files...")

    if not EMBEDDINGS_FILE.exists():
        raise FileNotFoundError(
            f"Embeddings file was not created:\n"
            f"{EMBEDDINGS_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file was not created:\n"
            f"{METADATA_FILE}"
        )

    saved_embeddings = np.load(
        EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    with METADATA_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_metadata = json.load(file)

    # --------------------------------------------------------
    # Count validation
    # --------------------------------------------------------

    if saved_embeddings.shape[0] != expected_count:
        raise ValueError(
            "Saved embedding count mismatch.\n"
            f"Expected: {expected_count:,}\n"
            f"Received: {saved_embeddings.shape[0]:,}"
        )

    if len(saved_metadata) != expected_count:
        raise ValueError(
            "Saved metadata count mismatch.\n"
            f"Expected: {expected_count:,}\n"
            f"Received: {len(saved_metadata):,}"
        )

    # --------------------------------------------------------
    # Dimension validation
    # --------------------------------------------------------

    if saved_embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            "Saved embedding dimension mismatch.\n"
            f"Expected: {EXPECTED_DIMENSION}\n"
            f"Received: {saved_embeddings.shape[1]}"
        )

    # --------------------------------------------------------
    # Alignment validation
    # --------------------------------------------------------

    for index in range(
        min(10, expected_count)
    ):
        metadata_index = saved_metadata[index][
            "vector_index"
        ]

        if metadata_index != index:
            raise ValueError(
                "Vector/metadata alignment error at "
                f"index {index}."
            )

    print("[OK] Embedding file verified.")
    print("[OK] Metadata file verified.")
    print("[OK] Vector count verified.")
    print("[OK] Dimension verified.")
    print("[OK] Vector/metadata alignment verified.")


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    embeddings: np.ndarray,
    metadata: list[dict[str, Any]],
) -> None:

    python_count = sum(
        1
        for item in metadata
        if item["metadata"]["technology"] == "python"
    )

    fastapi_count = sum(
        1
        for item in metadata
        if item["metadata"]["technology"] == "fastapi"
    )

    print()
    print_header("EMBEDDING STATISTICS")

    print(f"Total vectors       : {len(embeddings):,}")
    print(f"Embedding dimension : {embeddings.shape[1]}")
    print(f"Python vectors      : {python_count:,}")
    print(f"FastAPI vectors     : {fastapi_count:,}")
    print(f"Data type           : {embeddings.dtype}")
    print(
        f"Embeddings file     : "
        f"{EMBEDDINGS_FILE.stat().st_size / (1024 * 1024):.2f} MB"
    )
    print(
        f"Metadata file       : "
        f"{METADATA_FILE.stat().st_size / (1024 * 1024):.2f} MB"
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    print_header(
        "HYBRID RAG - EMBEDDING PIPELINE"
    )

    try:

        # ----------------------------------------------------
        # 1. Load chunks
        # ----------------------------------------------------

        print("[1/6] Loading chunks...")

        chunks = load_chunks()

        # ----------------------------------------------------
        # 2. Validate chunks
        # ----------------------------------------------------

        print()
        print("[2/6] Validating chunks...")

        validate_chunks(chunks)

        # ----------------------------------------------------
        # 3. Device
        # ----------------------------------------------------

        print()
        print("[3/6] Selecting device...")

        device = get_device()

        # ----------------------------------------------------
        # 4. Model
        # ----------------------------------------------------

        print()
        print("[4/6] Loading embedding model...")

        model = load_embedding_model(device)

        # ----------------------------------------------------
        # 5. Generate embeddings
        # ----------------------------------------------------

        print()
        print("[5/6] Generating embeddings...")

        embeddings = generate_embeddings(
            model,
            chunks,
        )

        metadata = build_metadata(
            chunks,
        )

        # ----------------------------------------------------
        # 6. Save + verify
        # ----------------------------------------------------

        print()
        print("[6/6] Saving and verifying...")

        save_vectorstore(
            embeddings,
            metadata,
        )

        verify_saved_files(
            expected_count=len(chunks),
        )

        print_statistics(
            embeddings,
            metadata,
        )

        print()
        print_header(
            "EMBEDDING PIPELINE COMPLETED SUCCESSFULLY"
        )

        print(
            f"Chunks processed : {len(chunks):,}"
        )

        print(
            f"Vectors created  : {len(embeddings):,}"
        )

        print(
            f"Dimension         : {embeddings.shape[1]}"
        )

        print(
            f"Vectorstore       : {VECTORSTORE_DIR}"
        )

        print()
        print("[OK] Ready for FAISS indexing.")

    except KeyboardInterrupt:

        print()
        print("[STOPPED] Embedding process interrupted by user.")
        sys.exit(130)

    except Exception as exc:

        print()
        print_header(
            "EMBEDDING PIPELINE FAILED"
        )

        print(f"[ERROR] {exc}")
        print()
        print(
            "No successful completion was recorded."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()