from __future__ import annotations

import sys
from pathlib import Path


# Add project root to Python import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.retrieval.hybrid import HybridRetriever
from src.generation.prompt_builder import build_prompt


def main():
    query = "How do I create dependencies in FastAPI?"

    print("[1] Loading hybrid retriever...")

    retriever = HybridRetriever(
        semantic_top_k=60,
        bm25_top_k=60,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
    )

    print("[2] Running hybrid retrieval...")

    retrieval = retriever.search(query)

    results = retrieval["hybrid_results"]

    print(f"[OK] Retrieved candidates: {len(results)}")

    if not results:
        raise RuntimeError(
            "Hybrid retrieval returned no results."
        )

    print("\n[3] Top 5 retrieved sources:")

    for result in results[:5]:
        metadata = result.get("metadata", {})

        print(
            f"  Rank {result['rank']} | "
            f"{metadata.get('technology', 'unknown')} | "
            f"{metadata.get('source', 'unknown')}"
        )

    print("\n[4] Building grounded prompt...")

    prompt = build_prompt(
        query=query,
        results=results,
        max_chunks=5,
    )

    print("\n" + "=" * 80)
    print(prompt)
    print("=" * 80)

    print(
        "\n[PASS] Retrieval -> Context -> Prompt pipeline works."
    )


if __name__ == "__main__":
    main()