from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
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

# GitHub Release artifact
DEFAULT_KB_URL = (
    "https://github.com/"
    "balasaikonka5851/"
    "hybrid-rag-technical-knowledge/"
    "releases/download/"
    "v1.0.0/"
    "hybridrag_knowledge_base.zip"
)

KB_URL_ENV = "KNOWLEDGE_BASE_URL"
KB_SHA256_ENV = "KNOWLEDGE_BASE_SHA256"


def _get_kb_url() -> str:
    """Return the configured knowledge-base download URL."""

    url = os.getenv(KB_URL_ENV, DEFAULT_KB_URL).strip()

    if not url:
        raise ValueError(
            "Knowledge-base URL cannot be empty."
        )

    if not url.startswith(("https://", "http://")):
        raise ValueError(
            "Knowledge-base URL must start with http:// or https://."
        )

    return url


def _get_expected_sha256() -> str | None:
    """Return optional SHA-256 checksum from environment."""

    value = os.getenv(KB_SHA256_ENV, "").strip()

    if not value:
        return None

    value = value.lower()

    if len(value) != 64:
        raise ValueError(
            "KNOWLEDGE_BASE_SHA256 must contain "
            "exactly 64 hexadecimal characters."
        )

    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(
            "KNOWLEDGE_BASE_SHA256 contains invalid "
            "hexadecimal characters."
        ) from error

    return value


def _calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _knowledge_base_exists() -> bool:
    """Return True when all required KB files exist."""

    required_files = (
        CHUNKS_PATH,
        EMBEDDINGS_PATH,
        FAISS_PATH,
        METADATA_PATH,
    )

    return all(path.exists() for path in required_files)


def download_artifact(
    url: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """
    Download the HybridRAG knowledge-base artifact.

    The artifact is downloaded to a temporary file first.
    It is never extracted directly from an unverified download.
    """

    url = url or _get_kb_url()

    if expected_sha256 is None:
        expected_sha256 = _get_expected_sha256()

    print("=" * 60)
    print("HYBRIDRAG KNOWLEDGE BASE DOWNLOAD")
    print("=" * 60)
    print(f"Source: {url}")

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="hybridrag_kb_"
        )
    )

    archive_path = (
        temp_dir / "hybridrag_knowledge_base.zip"
    )

    try:
        print("[1/3] Downloading artifact...")

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "HybridRAG/1.0",
            },
        )

        with urllib.request.urlopen(
            request,
            timeout=180,
        ) as response, archive_path.open("wb") as output:

            while True:
                chunk = response.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)

        if not archive_path.exists():
            raise FileNotFoundError(
                "Downloaded artifact was not created."
            )

        size_mb = (
            archive_path.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"[OK] Downloaded: "
            f"{size_mb:.2f} MiB"
        )

        print("[2/3] Verifying artifact...")

        actual_sha256 = _calculate_sha256(
            archive_path
        )

        print(
            f"[OK] SHA-256: "
            f"{actual_sha256}"
        )

        if expected_sha256:
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    "Knowledge-base artifact checksum "
                    "verification failed.\n"
                    f"Expected: {expected_sha256}\n"
                    f"Actual:   {actual_sha256}"
                )

            print(
                "[OK] SHA-256 checksum verified."
            )
        else:
            print(
                "[INFO] No SHA-256 checksum configured."
            )

        print("[3/3] Checking ZIP structure...")

        with zipfile.ZipFile(
            archive_path,
            "r",
        ) as archive:

            bad_member = archive.testzip()

            if bad_member is not None:
                raise ValueError(
                    "Corrupted ZIP member: "
                    f"{bad_member}"
                )

            members = archive.infolist()

            if not members:
                raise ValueError(
                    "Knowledge-base ZIP is empty."
                )

            print(
                f"[OK] ZIP contains "
                f"{len(members)} files."
            )

        print("[PASS] Artifact verified.")
        print("=" * 60)

        return archive_path

    except Exception:
        # Keep the error clear while removing
        # the temporary download.
        try:
            if archive_path.exists():
                archive_path.unlink()

            temp_dir.rmdir()
        except OSError:
            pass

        raise


def extract_artifact(
    zip_path: str | Path,
) -> None:
    """Extract a verified HybridRAG deployment artifact."""

    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Deployment artifact not found: "
            f"{zip_path}"
        )

    print("=" * 60)
    print("EXTRACTING HYBRIDRAG KNOWLEDGE BASE")
    print("=" * 60)

    root_resolved = ROOT.resolve()

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        # Prevent path traversal.
        for member in archive.infolist():

            target = (
                ROOT / member.filename
            ).resolve()

            try:
                target.relative_to(
                    root_resolved
                )
            except ValueError as error:
                raise ValueError(
                    "Unsafe path inside artifact: "
                    f"{member.filename}"
                ) from error

        archive.extractall(ROOT)

    print("[PASS] Artifact extracted.")
    print("=" * 60)


def ensure_knowledge_base() -> dict:
    """
    Ensure the local knowledge base exists.

    Local development:
        Use existing files.

    Cloud deployment:
        Download the GitHub Release artifact
        when the local knowledge base is missing.
    """

    if _knowledge_base_exists():
        print(
            "[OK] Local knowledge base already exists."
        )

        return validate_knowledge_base()

    print(
        "[INFO] Local knowledge base not found."
    )

    archive_path = download_artifact()

    try:
        extract_artifact(
            archive_path
        )
    finally:
        try:
            archive_path.unlink()
            archive_path.parent.rmdir()
        except OSError:
            pass

    print(
        "[OK] Downloaded knowledge base "
        "successfully."
    )

    return validate_knowledge_base()


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
            + "\n".join(
                f"  - {item}"
                for item in missing
            )
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    index = faiss.read_index(
        str(FAISS_PATH)
    )

    chunk_count = len(chunks)
    metadata_count = len(metadata)
    vector_count = embeddings.shape[0]

    if chunk_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Chunk count mismatch: "
            f"{chunk_count:,} != "
            f"{EXPECTED_CHUNKS:,}"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2-dimensional, "
            f"got {embeddings.ndim}"
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"{embeddings.shape[1]} != "
            f"{EXPECTED_DIMENSION}"
        )

    if index.ntotal != EXPECTED_CHUNKS:
        raise ValueError(
            f"FAISS vector count mismatch: "
            f"{index.ntotal:,} != "
            f"{EXPECTED_CHUNKS:,}"
        )

    if metadata_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Metadata count mismatch: "
            f"{metadata_count:,} != "
            f"{EXPECTED_CHUNKS:,}"
        )

    if vector_count != EXPECTED_CHUNKS:
        raise ValueError(
            f"Embedding count mismatch: "
            f"{vector_count:,} != "
            f"{EXPECTED_CHUNKS:,}"
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            "Embeddings contain NaN or "
            "infinite values."
        )

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
            "Embeddings are not normalized correctly."
        )

    return {
        "chunks": chunk_count,
        "embeddings": tuple(
            embeddings.shape
        ),
        "faiss_vectors": index.ntotal,
        "metadata": metadata_count,
        "dimension": embeddings.shape[1],
    }


def load_knowledge_base() -> dict:
    """
    Ensure and validate the knowledge base.
    """

    info = ensure_knowledge_base()

    print("=" * 60)
    print("HYBRIDRAG KNOWLEDGE BASE")
    print("=" * 60)

    print(
        f"Chunks:          "
        f"{info['chunks']:,}"
    )

    print(
        f"Embeddings:      "
        f"{info['embeddings']}"
    )

    print(
        f"FAISS vectors:   "
        f"{info['faiss_vectors']:,}"
    )

    print(
        f"Metadata:        "
        f"{info['metadata']:,}"
    )

    print(
        f"Dimensions:      "
        f"{info['dimension']}"
    )

    print(
        "[PASS] Knowledge base validated"
    )

    print("=" * 60)

    return info


if __name__ == "__main__":
    load_knowledge_base()