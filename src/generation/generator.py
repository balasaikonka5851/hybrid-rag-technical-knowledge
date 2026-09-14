from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from src.generation.prompt_builder import build_prompt


load_dotenv()


DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiGenerator:
    """
    Gemini-based answer generator for HybridRAG.

    Retrieval and generation are deliberately separated.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the .env file."
            )

        self.model = model

        self.client = genai.Client(
            api_key=api_key
        )

        print(
            f"[OK] Gemini generator initialized: "
            f"{self.model}"
        )

    def generate(
        self,
        query: str,
        results: list[dict[str, Any]],
        max_chunks: int = 5,
    ) -> str:

        prompt = build_prompt(
            query=query,
            results=results,
            max_chunks=max_chunks,
        )

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        answer = getattr(
            interaction,
            "output_text",
            None,
        )

        if not answer:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        return answer.strip()