"""
HybridRAG - Semantic Retrieval

Converts a user query into an embedding and searches
the FAISS vector index for semantically similar chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

FAISS_INDEX_FILE = VECTORSTORE_DIR / "faiss.index"
METADATA_FILE = VECTORSTORE_DIR / "metadata.json"


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_DIMENSION = 384


# ============================================================
# SEMANTIC RETRIEVER
# ============================================================

class SemanticRetriever:
    """
    Semantic retrieval using:

        Query
          ↓
        Embedding
          ↓
        FAISS
          ↓
        Top-K chunks
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ) -> None:

        self.model_name = model_name

        # ----------------------------------------------------
        # Validate files
        # ----------------------------------------------------

        if not FAISS_INDEX_FILE.exists():
            raise FileNotFoundError(
                f"FAISS index not found:\n"
                f"{FAISS_INDEX_FILE}\n\n"
                "Build the FAISS index first."
            )

        if not METADATA_FILE.exists():
            raise FileNotFoundError(
                f"Metadata file not found:\n"
                f"{METADATA_FILE}"
            )

        # ----------------------------------------------------
        # Load FAISS
        # ----------------------------------------------------

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(FAISS_INDEX_FILE)
        )

        if self.index.ntotal == 0:
            raise ValueError(
                "FAISS index contains zero vectors."
            )

        if self.index.d != EXPECTED_DIMENSION:
            raise ValueError(
                "FAISS dimension mismatch.\n"
                f"Expected: {EXPECTED_DIMENSION}\n"
                f"Received: {self.index.d}"
            )

        print(
            f"[OK] FAISS vectors: "
            f"{self.index.ntotal:,}"
        )

        # ----------------------------------------------------
        # Load metadata
        # ----------------------------------------------------

        print("Loading metadata...")

        with METADATA_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            self.metadata: list[dict[str, Any]] = json.load(file)

        if len(self.metadata) != self.index.ntotal:
            raise ValueError(
                "FAISS/metadata count mismatch.\n"
                f"FAISS:     {self.index.ntotal:,}\n"
                f"Metadata:  {len(self.metadata):,}"
            )

        print(
            f"[OK] Metadata records: "
            f"{len(self.metadata):,}"
        )

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print(
            f"Loading query embedding model:\n"
            f"  {self.model_name}"
        )

        self.model = SentenceTransformer(
            self.model_name
        )

        dimension = self.model.get_embedding_dimension()

        if dimension != EXPECTED_DIMENSION:
            raise ValueError(
                "Embedding model dimension mismatch.\n"
                f"Expected: {EXPECTED_DIMENSION}\n"
                f"Received: {dimension}"
            )

        print(
            f"[OK] Query embedding dimension: "
            f"{dimension}"
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar chunks.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.shape != (
            1,
            EXPECTED_DIMENSION,
        ):
            raise ValueError(
                "Unexpected query embedding shape:\n"
                f"{query_embedding.shape}"
            )

        # ----------------------------------------------------
        # FAISS search
        # ----------------------------------------------------

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        # ----------------------------------------------------
        # Build results
        # ----------------------------------------------------

        results = []

        for rank, (
            score,
            vector_index,
        ) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):

            if vector_index < 0:
                continue

            item = self.metadata[vector_index]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "vector_index": int(vector_index),
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "metadata": item["metadata"],
                }
            )

        return results


# ============================================================
# DISPLAY
# ============================================================

def display_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 80)

    print()
    print(f"Query: {query}")

    print()

    for result in results:

        metadata = result["metadata"]

        print("-" * 80)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"Score      : {result['score']:.4f}"
        )

        print(
            f"Technology : {metadata['technology']}"
        )

        print(
            f"Source     : {metadata['source']}"
        )

        print(
            f"Section    : {metadata['section']}"
        )

        print(
            f"Chunk ID   : {result['chunk_id']}"
        )

        print()

        # Display a manageable preview.
        text = result["text"]

        if len(text) > 500:
            text = text[:500] + "..."

        print(text)

    print()
    print("=" * 80)


# ============================================================
# INTERACTIVE MODE
# ============================================================

def main() -> None:

    print("=" * 80)
    print("HYBRID RAG - SEMANTIC RETRIEVAL")
    print("=" * 80)

    retriever = SemanticRetriever()

    print()
    print("Semantic retriever ready.")
    print()
    print("Try questions such as:")
    print("  1. How do I create dependencies in FastAPI?")
    print("  2. How does Python handle exceptions?")
    print("  3. How do I define a Pydantic model?")
    print()
    print("Type 'exit' to stop.")

    while True:

        try:

            query = input("\nQuery: ").strip()

        except (KeyboardInterrupt, EOFError):

            print()
            print("Exiting.")

            break

        if query.lower() in {
            "exit",
            "quit",
        }:
            print("Exiting.")
            break

        if not query:
            print(
                "[WARNING] Please enter a question."
            )
            continue

        try:

            results = retriever.search(
                query=query,
                top_k=5,
            )

            display_results(
                query,
                results,
            )

        except Exception as exc:

            print()
            print(
                f"[ERROR] Search failed: {exc}"
            )


if __name__ == "__main__":
    main()