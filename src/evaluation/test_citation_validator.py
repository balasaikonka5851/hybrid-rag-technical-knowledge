from __future__ import annotations

import sys
from pathlib import Path


# Add project root to Python path so this file
# can be executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.evaluation.citation_validator import (
    extract_citations,
    validate_citations,
)


def main():

    answer = """
    FastAPI allows dependencies to be defined using
    callable objects [SOURCE 1].

    Classes can also be used as dependencies [SOURCE 2].
    """

    sources = [
        {
            "source": "tutorial/dependencies.txt",
            "technology": "fastapi",
            "section": "Dependencies",
        },
        {
            "source": "tutorial/classes-as-dependencies.md",
            "technology": "fastapi",
            "section": "Classes as dependencies",
        },
    ]

    print("[1] Extracting citations...")

    citations = extract_citations(answer)

    print(f"[OK] Extracted citations: {citations}")

    print("\n[2] Validating citations...")

    result = validate_citations(
        answer=answer,
        sources=sources,
    )

    print("\nValidation result:")

    for key, value in result.items():
        print(f"  {key}: {value}")

    assert citations == [1, 2]

    assert result["has_citations"] is True

    assert result["all_citations_valid"] is True

    assert result["invalid_citation_count"] == 0

    assert result["valid_citation_count"] == 2

    print(
        "\n[PASS] Citation validation test completed."
    )


if __name__ == "__main__":
    main()