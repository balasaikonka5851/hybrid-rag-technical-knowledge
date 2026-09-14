from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.groundedness import (
    build_groundedness_prompt,
    parse_groundedness_response,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
QUERIES_FILE = PROJECT_ROOT / "data" / "evaluation" / "queries.json"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "groundedness_results.json"
)

# Start small. Change to None for all queries.
LIMIT = None


# ---------------------------------------------------------
# Load evaluation queries
# ---------------------------------------------------------
def load_queries():
    with open(QUERIES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


# ---------------------------------------------------------
# Evaluate one query
# ---------------------------------------------------------
def evaluate_query(pipeline, item, index):

    query_id = item["id"]
    query = item["query"]

    print("\n" + "=" * 80)
    print(f"[{index}] {query_id}")
    print(f"Query: {query}")
    print("=" * 80)

    start = time.perf_counter()

    # -----------------------------------------------------
    # RAG
    # -----------------------------------------------------
    result = pipeline.ask(query)

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # -----------------------------------------------------
    # Citation validation
    # -----------------------------------------------------
    citation_validation = result.get(
        "citation_validation",
        {},
    )

    # -----------------------------------------------------
    # Groundedness judge
    # -----------------------------------------------------
    prompt = build_groundedness_prompt(
        query=query,
        answer=answer,
        sources=sources,
    )

    judge_response = pipeline.generator.client.interactions.create(
        model=pipeline.generator.model,
        input=prompt,
    )

    judge_text = judge_response.output_text

    groundedness = parse_groundedness_response(
        judge_text
    )

    elapsed = time.perf_counter() - start

    print(f"\nAnswer preview:")
    print(answer[:500].replace("\n", " "))

    print("\nGroundedness:")
    print(f"  Grounded           : {groundedness['grounded']}")
    print(f"  Score              : {groundedness['score']:.3f}")
    print(
        f"  Supported claims   : "
        f"{groundedness['supported_claims']}"
    )
    print(
        f"  Unsupported claims : "
        f"{groundedness['unsupported_claims']}"
    )

    print("\nCitations:")
    print(
        f"  Valid              : "
        f"{citation_validation.get('all_citations_valid', False)}"
    )
    print(
        f"  Found              : "
        f"{citation_validation.get('citations_found', [])}"
    )

    print(f"\nTotal time: {elapsed:.2f}s")

    return {
        "id": query_id,
        "query": query,
        "technology": item.get("technology"),
        "answer": answer,
        "grounded": groundedness["grounded"],
        "groundedness_score": groundedness["score"],
        "supported_claims": groundedness["supported_claims"],
        "unsupported_claims": groundedness["unsupported_claims"],
        "groundedness_explanation": groundedness[
            "explanation"
        ],
        "citations_found": citation_validation.get(
            "citations_found",
            [],
        ),
        "citations_valid": citation_validation.get(
            "all_citations_valid",
            False,
        ),
        "retrieval_latency_ms": result.get(
            "retrieval_latency_ms",
            0,
        ),
        "generation_latency_ms": result.get(
            "generation_latency_ms",
            0,
        ),
        "total_latency_ms": result.get(
            "total_latency_ms",
            0,
        ),
        "source_count": len(sources),
    }


# ---------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------
def calculate_metrics(results):

    if not results:
        return {}

    count = len(results)

    grounded_count = sum(
        1 for r in results if r["grounded"]
    )

    citation_valid_count = sum(
        1 for r in results if r["citations_valid"]
    )

    avg_score = sum(
        r["groundedness_score"]
        for r in results
    ) / count

    avg_supported = sum(
        r["supported_claims"]
        for r in results
    ) / count

    avg_unsupported = sum(
        r["unsupported_claims"]
        for r in results
    ) / count

    avg_retrieval = sum(
        r["retrieval_latency_ms"]
        for r in results
    ) / count

    avg_generation = sum(
        r["generation_latency_ms"]
        for r in results
    ) / count

    avg_total = sum(
        r["total_latency_ms"]
        for r in results
    ) / count

    return {
        "queries_evaluated": count,
        "grounded_answers": grounded_count,
        "grounded_rate": grounded_count / count,
        "average_groundedness_score": avg_score,
        "average_supported_claims": avg_supported,
        "average_unsupported_claims": avg_unsupported,
        "citation_valid_rate": (
            citation_valid_count / count
        ),
        "average_retrieval_latency_ms": avg_retrieval,
        "average_generation_latency_ms": avg_generation,
        "average_total_latency_ms": avg_total,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("=" * 80)
    print("HYBRIDRAG GROUNDEDNESS BENCHMARK")
    print("=" * 80)

    queries = load_queries()

    if LIMIT is not None:
        queries = queries[:LIMIT]

    print(f"\nQueries selected: {len(queries)}")

    print("\n[1] Initializing RAG pipeline...")

    pipeline = RAGPipeline()

    print("\n[2] Running benchmark...")

    results = []

    for index, item in enumerate(
        queries,
        start=1,
    ):
        try:
            result = evaluate_query(
                pipeline,
                item,
                index,
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\n[ERROR] Query {item['id']} failed:"
            )
            print(exc)

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    metrics = calculate_metrics(results)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 80)

    for key, value in metrics.items():

        if isinstance(value, float):

            if "rate" in key or "score" in key:
                print(
                    f"{key:40}: "
                    f"{value:.3f}"
                )
            else:
                print(
                    f"{key:40}: "
                    f"{value:.2f}"
                )

        else:
            print(
                f"{key:40}: "
                f"{value}"
            )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    output = {
        "metrics": metrics,
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n[OK] Results saved to:"
        f"\n{OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # Quality gate
    # -----------------------------------------------------
    if results:

        grounded_rate = metrics["grounded_rate"]
        citation_rate = metrics["citation_valid_rate"]

        if grounded_rate >= 0.80 and citation_rate >= 0.80:
            print(
                "\n[PASS] Quality gate passed."
            )
        else:
            print(
                "\n[WARNING] Quality gate not passed."
            )

    print("\n[OK] Benchmark completed.")


if __name__ == "__main__":
    main()