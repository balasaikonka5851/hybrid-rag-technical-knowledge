from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.rag_pipeline import RAGPipeline


def main():

    pipeline = RAGPipeline(
        semantic_top_k=60,
        bm25_top_k=60,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
        context_chunks=5,
        model="gemini-3.6-flash",
    )

    result = pipeline.ask(
        "How do I create dependencies in FastAPI?"
    )

    print("\n" + "=" * 80)
    print("FINAL STRUCTURED RESULT")
    print("=" * 80)

    print("\nQUERY:")
    print(result["query"])

    print("\nCITATION VALIDATION:")

    validation = result["citation_validation"]

    print(
        f"  Citations found: "
        f"{validation['citations_found']}"
    )

    print(
        f"  Valid citations: "
        f"{validation['valid_citations']}"
    )

    print(
        f"  Invalid citations: "
        f"{validation['invalid_citations']}"
    )

    print(
        f"  All citations valid: "
        f"{validation['all_citations_valid']}"
    )

    print("\nSOURCES:")

    for source in result["sources"]:
        print(
            f"\n  Rank: {source['rank']}"
            f"\n  Technology: {source['technology']}"
            f"\n  Source: {source['source']}"
            f"\n  Section: {source['section']}"
            f"\n  Score: {source['score']}"
        )

    print("\nSTATISTICS:")
    print(
        f"  Retrieved candidates: "
        f"{result['retrieved_candidates']}"
    )
    print(
        f"  Context chunks: "
        f"{result['context_chunks']}"
    )
    print(
        f"  Model: "
        f"{result['model']}"
    )
    print(
        f"  Retrieval latency: "
        f"{result['retrieval_latency_ms']} ms"
    )
    print(
        f"  Generation latency: "
        f"{result['generation_latency_ms']} ms"
    )
    print(
        f"  Total latency: "
        f"{result['total_latency_ms']} ms"
    )

    print("\n" + "=" * 80)
    print("[PASS] Unified RAG pipeline test completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()