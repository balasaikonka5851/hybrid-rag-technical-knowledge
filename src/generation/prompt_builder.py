from __future__ import annotations

from typing import Any


SYSTEM_INSTRUCTION = """
You are a technical documentation assistant.

Answer the user's question using ONLY the provided documentation
context.

Rules:
1. Do not use information that is not supported by the context.
2. If the context does not contain enough information, say:
   "I couldn't find enough information in the provided documentation."
3. Do not invent APIs, parameters, classes, functions, or behavior.
4. Prefer concise, technically accurate explanations.
5. When useful, include short code examples based only on the context.
6. Cite the documentation sources used in the answer.
""".strip()


def build_context(
    results: list[dict[str, Any]],
    max_chunks: int = 5,
) -> str:

    if max_chunks <= 0:
        raise ValueError(
            "max_chunks must be greater than 0."
        )

    selected = results[:max_chunks]

    if not selected:
        return (
            "No relevant documentation context was retrieved."
        )

    sections: list[str] = []

    for index, result in enumerate(
        selected,
        start=1,
    ):

        metadata = result.get(
            "metadata",
            {},
        )

        technology = metadata.get(
            "technology",
            "unknown",
        )

        source = metadata.get(
            "source",
            "unknown",
        )

        section = metadata.get(
            "section",
            "",
        )

        text = str(
            result.get(
                "text",
                "",
            )
        ).strip()

        if not text:
            continue

        header = (
            f"[SOURCE {index}]\n"
            f"Technology: {technology}\n"
            f"Source: {source}"
        )

        if section:
            header += f"\nSection: {section}"

        sections.append(
            f"{header}\n"
            f"Content:\n{text}"
        )

    if not sections:
        return (
            "No usable documentation context "
            "was retrieved."
        )

    return "\n\n".join(sections)


def build_prompt(
    query: str,
    results: list[dict[str, Any]],
    max_chunks: int = 5,
) -> str:

    if not isinstance(query, str):
        raise TypeError(
            "query must be a string."
        )

    query = query.strip()

    if not query:
        raise ValueError(
            "query cannot be empty."
        )

    context = build_context(
        results,
        max_chunks=max_chunks,
    )

    return f"""
{SYSTEM_INSTRUCTION}

DOCUMENTATION CONTEXT
=====================

{context}

USER QUESTION
=============

{query}

ANSWER
======

Provide a concise answer grounded only in the documentation context.
""".strip()