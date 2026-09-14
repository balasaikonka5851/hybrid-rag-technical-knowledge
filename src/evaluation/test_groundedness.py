from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.groundedness import (
    build_groundedness_prompt,
    parse_groundedness_response,
)


def main():

    query = "How do I create dependencies in FastAPI?"

    print("[1] Initializing RAG pipeline...")

    pipeline = RAGPipeline(
        semantic_top_k=60,
        bm25_top_k=60,
        semantic_weight=0.75,
        bm25_weight=0.25,
        rrf_k=60,
        context_chunks=5,
        model="gemini-3.6-flash",
    )

    print("\n[2] Running RAG query...")

    result = pipeline.ask(query)

    answer = result["answer"]
    sources = result["sources"]

    print(
        f"\n[OK] Answer generated."
    )

    print(
        f"[OK] Evidence sources: "
        f"{len(sources)}"
    )

    print("\n[3] Building groundedness evaluation prompt...")

    prompt = build_groundedness_prompt(
        query=query,
        answer=answer,
        sources=sources,
    )

    print(
        "[OK] Groundedness prompt created."
    )

    print("\n[4] Calling Gemini as evaluation judge...")

    judge_response = pipeline.generator.client.interactions.create(
        model=pipeline.model,
        input=prompt,
    )

    response_text = getattr(
        judge_response,
        "output_text",
        None,
    )

    if not response_text:
        raise RuntimeError(
            "Groundedness judge returned an empty response."
        )

    print(
        "\n[OK] Judge response received."
    )

    print("\n[5] Parsing groundedness result...")

    evaluation = parse_groundedness_response(
        response_text
    )

    print("\n" + "=" * 80)
    print("GROUNDEDNESS EVALUATION")
    print("=" * 80)

    print(
        json.dumps(
            evaluation,
            indent=2,
        )
    )

    print("=" * 80)

    print("\n[PASS] Groundedness evaluation completed.")


if __name__ == "__main__":
    main()