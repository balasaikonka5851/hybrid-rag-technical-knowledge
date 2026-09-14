from __future__ import annotations

import json
import re
from typing import Any


def build_groundedness_prompt(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
) -> str:

    evidence_parts = []

    for index, source in enumerate(
        sources,
        start=1,
    ):
        evidence_parts.append(
            f"""
[SOURCE {index}]
Technology: {source.get("technology", "unknown")}
Source: {source.get("source", "unknown")}
Section: {source.get("section", "")}
Content:
{source.get("text", "")}
""".strip()
        )

    evidence = "\n\n".join(evidence_parts)

    return f"""
You are evaluating the groundedness of a technical
documentation answer.

QUESTION
========
{query}

RETRIEVED EVIDENCE
==================
{evidence}

GENERATED ANSWER
================
{answer}

TASK
====

Determine whether the generated answer is supported
by the retrieved evidence.

Evaluate the claims in the answer against the evidence.

A claim is supported only when the retrieved evidence
provides sufficient information for that claim.

Do not use outside knowledge.

Return ONLY valid JSON using exactly this structure:

{{
  "grounded": true,
  "score": 0.0,
  "supported_claims": 0,
  "unsupported_claims": 0,
  "explanation": "..."
}}

Rules:

- score must be between 0.0 and 1.0
- grounded should be true when the answer is sufficiently
  supported by the retrieved evidence
- grounded should be false when important claims are
  unsupported
- count meaningful factual claims
- do not penalize harmless wording differences
- do not use knowledge outside the provided evidence
""".strip()


def parse_groundedness_response(
    response_text: str,
) -> dict[str, Any]:

    if not isinstance(response_text, str):
        raise TypeError(
            "response_text must be a string."
        )

    text = response_text.strip()

    # Handle accidental markdown code fences.
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        result = json.loads(text)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Groundedness judge did not return valid JSON."
        ) from exc

    required_fields = {
        "grounded",
        "score",
        "supported_claims",
        "unsupported_claims",
        "explanation",
    }

    missing = required_fields - result.keys()

    if missing:
        raise ValueError(
            f"Groundedness result is missing fields: "
            f"{sorted(missing)}"
        )

    score = float(result["score"])

    if not 0.0 <= score <= 1.0:
        raise ValueError(
            "Groundedness score must be between 0 and 1."
        )

    result["grounded"] = bool(
        result["grounded"]
    )

    result["score"] = score

    result["supported_claims"] = int(
        result["supported_claims"]
    )

    result["unsupported_claims"] = int(
        result["unsupported_claims"]
    )

    result["explanation"] = str(
        result["explanation"]
    )

    return result