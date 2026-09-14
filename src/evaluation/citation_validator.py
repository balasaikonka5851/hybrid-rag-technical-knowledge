from __future__ import annotations

import re
from typing import Any


SOURCE_PATTERN = re.compile(
    r"\[SOURCE\s+(\d+)\]",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[int]:
    """
    Extract source numbers cited in an answer.

    Example:
        [SOURCE 1, SOURCE 3]

    Returns:
        [1, 3]
    """

    if not isinstance(answer, str):
        raise TypeError(
            "answer must be a string."
        )

    matches = SOURCE_PATTERN.findall(answer)

    citations = []

    for match in matches:
        number = int(match)

        if number not in citations:
            citations.append(number)

    return citations


def validate_citations(
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate citations against retrieved sources.

    Source numbers are 1-based and correspond to
    the ordering supplied to the generator.
    """

    citations = extract_citations(answer)

    available_sources = set(
        range(
            1,
            len(sources) + 1,
        )
    )

    valid_citations = [
        citation
        for citation in citations
        if citation in available_sources
    ]

    invalid_citations = [
        citation
        for citation in citations
        if citation not in available_sources
    ]

    cited_sources = []

    for citation in valid_citations:

        source = sources[citation - 1]

        cited_sources.append(
            {
                "citation": citation,
                "source": source.get(
                    "source",
                    "unknown",
                ),
                "technology": source.get(
                    "technology",
                    "unknown",
                ),
                "section": source.get(
                    "section",
                    "",
                ),
            }
        )

    return {
        "citations_found": citations,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "citation_count": len(citations),
        "valid_citation_count": len(
            valid_citations
        ),
        "invalid_citation_count": len(
            invalid_citations
        ),
        "has_citations": bool(citations),
        "all_citations_valid": (
            len(invalid_citations) == 0
        ),
        "cited_sources": cited_sources,
    }