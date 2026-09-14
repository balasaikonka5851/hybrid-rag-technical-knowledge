from __future__ import annotations

import sys
from pathlib import Path


# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.hybrid import HybridRetriever
from src.generation.generator import GeminiGenerator


def main():

    query = "How do I create dependencies in FastAPI?"

    print("[1] Initializing hybrid retriever...")

    retriever = HybridRetriever(
        semantic_top_k=60,
        bm25_top_k=60,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
    )

    print("\n[2] Running hybrid retrieval...")

    retrieval = retriever.search(query)

    results = retrieval["hybrid_results"]

    if not results:
        raise RuntimeError(
            "No retrieval results found."
        )

    print(
        f"[OK] Retrieved {len(results)} candidates."
    )

    print("\n[3] Initializing Gemini...")

    generator = GeminiGenerator()

    print("\n[4] Generating grounded answer...")

    answer = generator.generate(
        query=query,
        results=results,
        max_chunks=5,
    )

    print("\n" + "=" * 80)
    print("FINAL ANSWER")
    print("=" * 80)

    print(answer)

    print("=" * 80)

    print("\n[PASS] End-to-end RAG generation completed.")


if __name__ == "__main__":
    main()