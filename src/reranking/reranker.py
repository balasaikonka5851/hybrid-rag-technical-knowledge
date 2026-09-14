from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """
    Cross-encoder reranker for reranking retrieved RAG candidates.

    The model jointly evaluates:
        (query, document)

    rather than embedding query and document independently.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:

        if not model_name.strip():
            raise ValueError("model_name cannot be empty.")

        if max_length <= 0:
            raise ValueError("max_length must be greater than 0.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")

        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print("Loading cross-encoder tokenizer:")
        print(f"  {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        print("Loading cross-encoder model...")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"[OK] Cross-encoder device: {self.device}")
        print(f"[OK] Cross-encoder model: {self.model_name}")

    @torch.inference_mode()
    def predict(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not documents:
            return []

        cleaned_documents = []

        for document in documents:
            if not isinstance(document, str):
                raise TypeError("Every document must be a string.")

            cleaned_documents.append(document.strip())

        scores: list[float] = []

        for start in range(0, len(cleaned_documents), self.batch_size):

            batch_documents = cleaned_documents[
                start : start + self.batch_size
            ]

            queries = [query] * len(batch_documents)

            encoded = self.tokenizer(
                queries,
                batch_documents,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)

            logits = outputs.logits

            # MS MARCO cross-encoder has one relevance score.
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]

            # Defensive handling for models with multiple logits.
            elif logits.ndim == 2:
                batch_scores = logits[:, -1]

            else:
                raise RuntimeError(
                    f"Unexpected model output shape: {tuple(logits.shape)}"
                )

            scores.extend(
                float(score.detach().cpu().item())
                for score in batch_scores
            )

        if len(scores) != len(documents):
            raise RuntimeError(
                "Number of reranker scores does not match "
                "number of documents."
            )

        return scores

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if not results:
            return []

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        documents = []

        for result in results:

            if "text" not in result:
                raise KeyError(
                    "Every retrieval result must contain a 'text' field."
                )

            documents.append(str(result["text"]))

        scores = self.predict(
            query=query,
            documents=documents,
        )

        reranked = []

        for original_rank, (result, score) in enumerate(
            zip(results, scores),
            start=1,
        ):

            item = dict(result)

            item["reranker_score"] = score
            item["original_rank"] = original_rank

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        reranked = reranked[:top_k]

        for rank, item in enumerate(reranked, start=1):
            item["rerank_rank"] = rank

        return reranked


def _demo() -> None:

    print("=" * 100)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 100)

    reranker = CrossEncoderReranker()

    query = "How do I create dependencies in FastAPI?"

    documents = [
        (
            "FastAPI supports dependencies that can be declared using "
            "the Depends function."
        ),
        (
            "Python exceptions can be handled with try and except "
            "statements."
        ),
        (
            "FastAPI dependencies allow common logic to be shared "
            "between path operations."
        ),
    ]

    scores = reranker.predict(
        query=query,
        documents=documents,
    )

    print()
    print("Query:")
    print(query)

    print()
    print("Scores:")

    for index, (document, score) in enumerate(
        zip(documents, scores),
        start=1,
    ):
        print(f"{index}. {score:.6f}")
        print(f"   {document}")

    print()
    print("[OK] Cross-encoder test completed.")


if __name__ == "__main__":
    _demo()