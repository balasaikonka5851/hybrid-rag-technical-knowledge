"""
HYBRID RAG - PYTHON BENCHMARK QUALITY AUDITOR

Audits the Python evaluation benchmark against the actual processed
corpus source representation.

Benchmark source format:
    tutorial/errors.txt
    library/exceptions.txt

Corpus source format:
    data/raw/python/tutorial/errors.txt
    data/raw/python/library/exceptions.txt

This script is READ-ONLY.
It never modifies queries.json or chunks.json.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks.json"
)

QUERIES_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "queries.json"
)


# =====================================================================
# DIAGNOSTIC CONCEPT HINTS
# =====================================================================

CONCEPT_HINTS: dict[str, list[str]] = {

    "q25": [
        "exception",
        "try",
        "except",
        "raise",
        "finally",
    ],

    "q26": [
        "except",
        "exception",
        "catch",
    ],

    "q27": [
        "raise",
        "exception",
    ],

    "q28": [
        "finally",
        "exception",
    ],

    "q29": [
        "try",
        "except",
    ],

    "q30": [
        "ValueError",
        "value error",
    ],

    "q31": [
        "function",
        "def ",
        "parameter",
        "argument",
    ],

    "q32": [
        "default argument",
        "default parameter",
        "argument",
        "parameter",
    ],

    "q33": [
        "class",
        "object",
        "instance",
    ],

    "q34": [
        "inheritance",
        "base class",
        "derived class",
        "subclass",
    ],

    "q35": [
        "import",
        "module",
        "importing",
    ],

    "q36": [
        "package",
        "packages",
        "__init__.py",
    ],

    "q37": [
        "async",
        "await",
        "asyncio",
        "coroutine",
    ],

    "q38": [
        "generator",
        "generators",
        "yield",
        "iterator",
    ],

    "q39": [
        "context manager",
        "context managers",
        "with ",
        "__enter__",
        "__exit__",
    ],

    "q40": [
        "list comprehension",
        "list comprehensions",
        "comprehension",
    ],
}


# =====================================================================
# NORMALIZATION
# =====================================================================

def normalize_path(value: Any) -> str:
    """
    Normalize path separators and casing.

    Examples:

        tutorial/errors.txt
        ./tutorial/errors.txt
        python/tutorial/errors.txt
        data/raw/python/tutorial/errors.txt

    become slash-normalized lowercase strings.
    """

    if value is None:
        return ""

    text = str(value).strip().replace("\\", "/")

    while text.startswith("./"):
        text = text[2:]

    return text.lower()


def canonical_python_source(
    benchmark_source: str,
) -> str:
    """
    Convert benchmark-relative source into the exact logical
    Python source identity.

    Example:

        tutorial/errors.txt
        ->
        data/raw/python/tutorial/errors.txt
    """

    source = normalize_path(benchmark_source)

    # Already canonical.
    if source.startswith("data/raw/python/"):
        return source

    # Remove an accidental python/ prefix.
    if source.startswith("python/"):
        source = source[len("python/"):]

    return f"data/raw/python/{source}"


def benchmark_source_from_corpus_source(
    corpus_source: str,
) -> str:
    """
    Reverse mapping used for diagnostics.

    Example:

        data/raw/python/tutorial/errors.txt
        ->
        tutorial/errors.txt
    """

    source = normalize_path(corpus_source)

    prefix = "data/raw/python/"

    if source.startswith(prefix):
        return source[len(prefix):]

    return source


# =====================================================================
# CHUNK METADATA
# =====================================================================

def get_metadata(
    chunk: dict[str, Any],
) -> dict[str, Any]:

    metadata = chunk.get("metadata", {})

    if isinstance(metadata, dict):
        return metadata

    return {}


def get_source(
    chunk: dict[str, Any],
) -> str:

    metadata = get_metadata(chunk)

    source = metadata.get("source")

    if source:
        return normalize_path(source)

    return ""


def get_technology(
    chunk: dict[str, Any],
) -> str:

    metadata = get_metadata(chunk)

    technology = metadata.get("technology")

    if technology:
        return str(technology).strip().lower()

    source = get_source(chunk)

    if source.startswith("data/raw/python/"):
        return "python"

    if source.startswith("data/raw/fastapi/"):
        return "fastapi"

    return "unknown"


def get_text(
    chunk: dict[str, Any],
) -> str:

    value = chunk.get("text", "")

    if value is None:
        return ""

    return str(value)


def get_chunk_id(
    chunk: dict[str, Any],
) -> str:

    value = chunk.get("id")

    if value:
        return str(value)

    value = chunk.get("chunk_id")

    if value:
        return str(value)

    return ""


# =====================================================================
# LOADING
# =====================================================================

def load_json(
    path: Path,
) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# =====================================================================
# CONCEPT MATCHING
# =====================================================================

def find_concept_hits(
    text: str,
    hints: list[str],
) -> list[str]:

    lowered = text.lower()

    return [
        hint
        for hint in hints
        if hint.lower() in lowered
    ]


# =====================================================================
# CORPUS INDEX
# =====================================================================

def build_indexes(
    chunks: list[dict[str, Any]],
):

    source_to_chunks: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    source_technologies: dict[
        str,
        Counter[str],
    ] = defaultdict(Counter)

    for chunk in chunks:

        source = get_source(chunk)

        if not source:
            continue

        technology = get_technology(chunk)

        source_to_chunks[source].append(chunk)

        source_technologies[source][technology] += 1

    return (
        source_to_chunks,
        source_technologies,
    )


# =====================================================================
# EVIDENCE DISPLAY
# =====================================================================

def print_evidence(
    evidence: list[dict[str, Any]],
    limit: int = 2,
) -> None:

    for item in evidence[:limit]:

        snippet = re.sub(
            r"\s+",
            " ",
            item["text"],
        ).strip()

        if len(snippet) > 260:
            snippet = snippet[:260] + "..."

        print(
            f"      chunk={item['chunk_id']}"
        )

        print(
            f"      concept hits={item['hits']}"
        )

        print(
            f"      {snippet}"
        )


# =====================================================================
# MAIN AUDIT
# =====================================================================

def main() -> None:

    print("=" * 110)
    print(
        "HYBRID RAG - PYTHON BENCHMARK QUALITY AUDIT"
    )
    print("=" * 110)

    chunks = load_json(CHUNKS_FILE)
    queries = load_json(QUERIES_FILE)

    if not isinstance(chunks, list):
        raise ValueError(
            "chunks.json must contain a list."
        )

    if not isinstance(queries, list):
        raise ValueError(
            "queries.json must contain a list."
        )

    (
        source_to_chunks,
        source_technologies,
    ) = build_indexes(chunks)

    print(
        f"[INFO] Corpus chunks: {len(chunks):,}"
    )

    print(
        f"[INFO] Evaluation queries: {len(queries):,}"
    )

    # ---------------------------------------------------------------
    # Technology counts
    # ---------------------------------------------------------------

    technology_counts = Counter(
        get_technology(chunk)
        for chunk in chunks
    )

    print("\n[CORPUS TECHNOLOGY COUNTS]")

    for technology, count in technology_counts.most_common():

        print(
            f"  {technology:<12} {count:>8,}"
        )

    # ---------------------------------------------------------------
    # Python queries
    # ---------------------------------------------------------------

    python_queries = [
        query
        for query in queries
        if str(
            query.get(
                "technology",
                "",
            )
        ).strip().lower() == "python"
    ]

    print(
        f"\n[INFO] Python queries audited: "
        f"{len(python_queries)}"
    )

    # ---------------------------------------------------------------
    # Audit counters
    # ---------------------------------------------------------------

    passed = 0
    review = 0
    failed = 0

    print("\n[PYTHON QUERY AUDIT]")
    print("-" * 110)

    # ===============================================================
    # EACH QUERY
    # ===============================================================

    for query in python_queries:

        qid = str(
            query.get(
                "id",
                "UNKNOWN",
            )
        )

        question = str(
            query.get(
                "query",
                "",
            )
        ).strip()

        benchmark_sources = [
            normalize_path(source)
            for source in query.get(
                "relevant_sources",
                [],
            )
        ]

        print(
            f"\n{qid}: {question}"
        )

        print(
            f"  Ground truth: {benchmark_sources}"
        )

        # -----------------------------------------------------------
        # Schema validation
        # -----------------------------------------------------------

        if not benchmark_sources:

            print(
                "  [FAIL] relevant_sources is empty."
            )

            failed += 1
            continue

        # -----------------------------------------------------------
        # Resolve benchmark source -> corpus source
        # -----------------------------------------------------------

        resolved_sources = []

        missing_sources = []

        for benchmark_source in benchmark_sources:

            canonical = canonical_python_source(
                benchmark_source
            )

            if canonical in source_to_chunks:

                resolved_sources.append(
                    canonical
                )

            else:

                missing_sources.append(
                    benchmark_source
                )

        if missing_sources:

            print(
                "  [FAIL] Could not map source(s):"
            )

            for source in missing_sources:

                print(
                    f"      benchmark: {source}"
                )

                print(
                    f"      expected:  "
                    f"{canonical_python_source(source)}"
                )

            failed += 1
            continue

        print(
            "  [OK] Ground-truth source(s) exist."
        )

        # -----------------------------------------------------------
        # Technology validation
        # -----------------------------------------------------------

        wrong_technology = []

        for source in resolved_sources:

            technologies = source_technologies[
                source
            ]

            if "python" not in technologies:

                wrong_technology.append(
                    (
                        source,
                        dict(technologies),
                    )
                )

        if wrong_technology:

            print(
                "  [FAIL] Ground-truth source is "
                "not Python."
            )

            for source, technologies in wrong_technology:

                print(
                    f"      {source}: "
                    f"{technologies}"
                )

            failed += 1
            continue

        print(
            "  [OK] Ground-truth source technology = Python."
        )

        # -----------------------------------------------------------
        # Concept evidence inside GT sources
        # -----------------------------------------------------------

        hints = CONCEPT_HINTS.get(
            qid,
            [],
        )

        gt_evidence = []

        for source in resolved_sources:

            for chunk in source_to_chunks[source]:

                text = get_text(chunk)

                hits = find_concept_hits(
                    text,
                    hints,
                )

                if hits:

                    gt_evidence.append(
                        {
                            "source": source,
                            "chunk_id": get_chunk_id(chunk),
                            "hits": hits,
                            "text": text,
                        }
                    )

        if gt_evidence:

            print(
                f"  [OK] Concept evidence found in "
                f"{len(gt_evidence)} GT chunk(s)."
            )

            print_evidence(
                gt_evidence
            )

        else:

            print(
                "  [REVIEW] No obvious concept evidence "
                "found in GT source."
            )

            review += 1
            continue

        # -----------------------------------------------------------
        # Search Python corpus for competing evidence
        # -----------------------------------------------------------

        competing_sources: Counter[str] = Counter()

        for source, source_chunks in source_to_chunks.items():

            # Only Python sources.
            if "python" not in source_technologies[source]:
                continue

            # Skip GT sources.
            if source in resolved_sources:
                continue

            matching_chunks = 0

            for chunk in source_chunks:

                text = get_text(chunk)

                hits = find_concept_hits(
                    text,
                    hints,
                )

                if hits:
                    matching_chunks += 1

            if matching_chunks:
                competing_sources[
                    benchmark_source_from_corpus_source(
                        source
                    )
                ] = matching_chunks

        if competing_sources:

            print(
                "  [INFO] Other Python sources also "
                "contain concept terms:"
            )

            for source, count in (
                competing_sources.most_common(5)
            ):

                print(
                    f"      {source} "
                    f"({count} matching chunk(s))"
                )

        # -----------------------------------------------------------
        # PASS
        # -----------------------------------------------------------

        passed += 1

        print(
            "  [PASS] Ground-truth mapping has "
            "direct concept evidence."
        )

    # ===============================================================
    # SUMMARY
    # ===============================================================

    print("\n" + "=" * 110)
    print("AUDIT SUMMARY")
    print("=" * 110)

    print(
        f"Python queries : {len(python_queries)}"
    )

    print(
        f"PASS           : {passed}"
    )

    print(
        f"REVIEW         : {review}"
    )

    print(
        f"FAIL           : {failed}"
    )

    print("=" * 110)

    if failed:

        print(
            "[ACTION] Investigate source mapping failures."
        )

    elif review:

        print(
            "[ACTION] Manually inspect REVIEW queries."
        )

    else:

        print(
            "[OK] Python benchmark passed the "
            "structural/evidence audit."
        )

    print(
        "\n[IMPORTANT] "
        "No benchmark or corpus files were modified."
    )


if __name__ == "__main__":
    main()