"""
HybridRAG - BM25 Keyword Retrieval

Provides lexical/keyword-based retrieval over the same chunks
used by the semantic FAISS retriever.

Input:
    data/processed/chunks.json

BM25 is useful for:
    - exact technical terms
    - class/function names
    - error names
    - configuration names
    - API terminology
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_TOP_K = 5


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.

    We intentionally preserve technical identifiers reasonably
    well while also handling punctuation.

    Examples:

        RequestValidationError
        FastAPI
        HTTPException
        response_model
        async def

    become searchable tokens.
    """

    if not isinstance(text, str):
        return []

    text = text.lower()

    # Keep alphanumeric characters and underscores.
    tokens = re.findall(
        r"[a-zA-Z0-9_]+",
        text,
    )

    return tokens


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks() -> list[dict[str, Any]]:
    """
    Load chunks.json and validate the basic structure.
    """

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"chunks.json not found:\n"
            f"{CHUNKS_FILE}\n\n"
            "Run the chunking pipeline first."
        )

    print("Loading chunks...")
    print(f"  {CHUNKS_FILE}")

    try:
        with CHUNKS_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            chunks = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in chunks.json:\n{exc}"
        ) from exc

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json must contain a list."
        )

    if not chunks:
        raise ValueError(
            "chunks.json is empty."
        )

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} is not a dictionary."
            )

        if "id" not in chunk:
            raise ValueError(
                f"Chunk {index} has no id."
            )

        if "text" not in chunk:
            raise ValueError(
                f"Chunk {index} has no text."
            )

        if "metadata" not in chunk:
            raise ValueError(
                f"Chunk {index} has no metadata."
            )

    print(
        f"[OK] Loaded {len(chunks):,} chunks."
    )

    return chunks


# ============================================================
# BM25 RETRIEVER
# ============================================================

class BM25Retriever:
    """
    Keyword retrieval using BM25Okapi.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]],
    ) -> None:

        if not chunks:
            raise ValueError(
                "Cannot create BM25 retriever "
                "with zero chunks."
            )

        self.chunks = chunks

        print()
        print("Tokenizing documents...")

        self.tokenized_corpus = [
            tokenize(chunk["text"])
            for chunk in chunks
        ]

        # Detect completely empty documents.
        empty_count = sum(
            1
            for tokens in self.tokenized_corpus
            if not tokens
        )

        if empty_count > 0:
            raise ValueError(
                f"{empty_count} chunks produced "
                "empty token lists."
            )

        print(
            f"[OK] Tokenized "
            f"{len(self.tokenized_corpus):,} chunks."
        )

        print()
        print("Building BM25 index...")

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("[OK] BM25 index created.")

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict[str, Any]]:
        """
        Search the corpus using BM25.
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
            len(self.chunks),
        )

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        scores = self.bm25.get_scores(
            query_tokens
        )

        # Get top-k indexes efficiently.
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results = []

        for rank, index in enumerate(
            ranked_indices,
            start=1,
        ):

            chunk = self.chunks[index]

            results.append(
                {
                    "rank": rank,
                    "score": float(scores[index]),
                    "vector_index": index,
                    "chunk_id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                }
            )

        return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query: str,
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 80)
    print("BM25 SEARCH RESULTS")
    print("=" * 80)

    print()
    print(f"Query: {query}")

    if not results:
        print()
        print("[INFO] No keyword matches found.")
        return

    for result in results:

        metadata = result["metadata"]

        print()
        print("-" * 80)

        print(
            f"Rank       : {result['rank']}"
        )

        print(
            f"BM25 score : {result['score']:.4f}"
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

        text = result["text"]

        if len(text) > 500:
            text = text[:500] + "..."

        print(text)

    print()
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("HYBRID RAG - BM25 KEYWORD RETRIEVAL")
    print("=" * 80)

    try:

        # ----------------------------------------------------
        # Load corpus
        # ----------------------------------------------------

        chunks = load_chunks()

        # ----------------------------------------------------
        # Build BM25
        # ----------------------------------------------------

        retriever = BM25Retriever(
            chunks
        )

        print()
        print("BM25 retriever ready.")

        print()
        print("Try exact technical queries such as:")
        print(
            "  1. RequestValidationError"
        )
        print(
            "  2. HTTPException FastAPI"
        )
        print(
            "  3. response_model"
        )
        print(
            "  4. Python ValueError"
        )

        print()
        print("Type 'exit' to stop.")

        # ----------------------------------------------------
        # Interactive search
        # ----------------------------------------------------

        while True:

            try:

                query = input(
                    "\nQuery: "
                ).strip()

            except (
                KeyboardInterrupt,
                EOFError,
            ):

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
                    "[WARNING] "
                    "Please enter a question."
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

    except Exception as exc:

        print()
        print("=" * 80)
        print("BM25 RETRIEVER FAILED")
        print("=" * 80)

        print()
        print(f"[ERROR] {exc}")

        sys.exit(1)


if __name__ == "__main__":
    main()