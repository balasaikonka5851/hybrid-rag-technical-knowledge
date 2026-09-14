from __future__ import annotations

import json
import zipfile
from pathlib import Path

import faiss
import numpy as np


ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
EMBEDDINGS_PATH = ROOT / "vectorstore" / "embeddings.npy"
FAISS_PATH = ROOT / "vectorstore" / "faiss.index"
METADATA_PATH = ROOT / "vectorstore" / "metadata.json"

EXPECTED_CHUNKS = 18_150
EXPECTED_DIMENSION = 384


def validate_knowledge_base() -> dict:
    """Validate the complete local RAG knowledge base."""

    required_files = {
        "chunks": CHUNKS_PATH,
        "embeddings": EMBEDDINGS_PATH,
        "faiss": FAISS_PATH,
        "metadata": METADATA_PATH,
    }

    missing = [
        str(path.relative_to(ROOT))
        for path in required_files.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing RAG knowledge-base files:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )

    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(str(FAISS_PATH))

    chunk_count = len(chunks)
    metadata_count = len(metadata)
    vector_count = embeddings.shape[0]

    if chunk_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Chunk count mismatch: "
            f"{chunk_count:,} != {EXPECTED_CHUNKS:,}"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2-dimensional, got {embeddings.ndim}"
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"{embeddings.shape[1]} != {EXPECTED_DIMENSION}"
        )

    if index.ntotal != EXPECTED_CHUNKS:
        raise ValueError(
            f"FAISS vector count mismatch: "
            f"{index.ntotal:,} != {EXPECTED_CHUNKS:,}"
        )

    if metadata_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Metadata count mismatch: "
            f"{metadata_count:,} != {EXPECTED_CHUNKS:,}"
        )

    if vector_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Embedding count mismatch: "
            f"{vector_count:,} != {EXPECTED_CHUNKS:,}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            "Embeddings are not normalized correctly."
        )

    return {
        "chunks": chunk_count,
        "embeddings": tuple(embeddings.shape),
        "faiss_vectors": index.ntotal,
        "metadata": metadata_count,
        "dimension": embeddings.shape[1],
    }


def extract_artifact(zip_path: str | Path) -> None:
    """Extract a HybridRAG deployment artifact."""

    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Deployment artifact not found: {zip_path}"
        )

    with zipfile.ZipFile(zip_path, "r") as archive:

        # Prevent path traversal when extracting archives.
        for member in archive.infolist():
            target = (ROOT / member.filename).resolve()

            if not str(target).startswith(str(ROOT.resolve())):
                raise ValueError(
                    f"Unsafe path inside artifact: {member.filename}"
                )

        archive.extractall(ROOT)


def load_knowledge_base() -> dict:
    """
    Validate the knowledge base and return its metadata.

    This function intentionally does not silently rebuild indexes.
    A corrupted or incomplete deployment should fail clearly.
    """

    info = validate_knowledge_base()

    print("=" * 60)
    print("HYBRIDRAG KNOWLEDGE BASE")
    print("=" * 60)
    print(f"Chunks:          {info['chunks']:,}")
    print(f"Embeddings:      {info['embeddings']}")
    print(f"FAISS vectors:   {info['faiss_vectors']:,}")
    print(f"Metadata:        {info['metadata']:,}")
    print(f"Dimensions:      {info['dimension']}")
    print("[PASS] Knowledge base validated")
    print("=" * 60)

    return info


if __name__ == "__main__":
    load_knowledge_base()