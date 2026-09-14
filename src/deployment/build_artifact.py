from pathlib import Path
import zipfile
import json
import numpy as np
import faiss


ROOT = Path(__file__).resolve().parents[2]

CHUNKS = ROOT / "data" / "processed" / "chunks.json"
EMBEDDINGS = ROOT / "vectorstore" / "embeddings.npy"
FAISS_INDEX = ROOT / "vectorstore" / "faiss.index"
METADATA = ROOT / "vectorstore" / "metadata.json"

OUTPUT = ROOT / "hybridrag_knowledge_base.zip"

EXPECTED_CHUNKS = 18_150
EXPECTED_DIM = 384


def validate():
    print("=" * 60)
    print("HYBRIDRAG DEPLOYMENT ARTIFACT VALIDATION")
    print("=" * 60)

    files = [CHUNKS, EMBEDDINGS, FAISS_INDEX, METADATA]

    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"[OK] {path.relative_to(ROOT)}")

    with open(CHUNKS, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS)

    index = faiss.read_index(str(FAISS_INDEX))

    with open(METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print()
    print(f"Chunks:       {len(chunks):,}")
    print(f"Embeddings:   {embeddings.shape}")
    print(f"FAISS:        {index.ntotal:,} vectors")
    print(f"Metadata:     {len(metadata):,}")

    if len(chunks) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS:,} chunks, found {len(chunks):,}"
        )

    if embeddings.shape != (EXPECTED_CHUNKS, EXPECTED_DIM):
        raise ValueError(
            f"Unexpected embeddings shape: {embeddings.shape}"
        )

    if index.ntotal != EXPECTED_CHUNKS:
        raise ValueError(
            f"FAISS contains {index.ntotal:,} vectors, "
            f"expected {EXPECTED_CHUNKS:,}"
        )

    if len(metadata) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Metadata contains {len(metadata):,} records, "
            f"expected {EXPECTED_CHUNKS:,}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings are not properly normalized")

    print()
    print("[PASS] Chunk count validated")
    print("[PASS] Embedding dimensions validated")
    print("[PASS] FAISS vector count validated")
    print("[PASS] Metadata alignment validated")
    print("[PASS] Embedding numerical validity validated")
    print("[PASS] Embedding normalization validated")

    return True


def build_zip():
    if OUTPUT.exists():
        OUTPUT.unlink()

    print()
    print("Creating deployment artifact...")

    with zipfile.ZipFile(
        OUTPUT,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as z:

        z.write(CHUNKS, "data/processed/chunks.json")
        z.write(EMBEDDINGS, "vectorstore/embeddings.npy")
        z.write(FAISS_INDEX, "vectorstore/faiss.index")
        z.write(METADATA, "vectorstore/metadata.json")

    size_mb = OUTPUT.stat().st_size / (1024 * 1024)

    print(f"[OK] Artifact created: {OUTPUT.name}")
    print(f"[OK] Size: {size_mb:.2f} MiB")


if __name__ == "__main__":
    validate()
    build_zip()

    print()
    print("=" * 60)
    print("DEPLOYMENT ARTIFACT READY")
    print("=" * 60)