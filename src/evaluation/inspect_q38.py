import json
import re
from pathlib import Path


CHUNKS_PATH = Path("data/processed/chunks.json")
TARGET_SOURCE = "data/raw/python/tutorial/classes.txt"


def normalize_source(source: str) -> str:
    return source.replace("\\", "/").strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

    print("=" * 100)
    print("PYTHON tutorial/classes.txt - GENERATOR EVIDENCE")
    print("=" * 100)

    matches = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        technology = metadata.get("technology", "")
        source = normalize_source(metadata.get("source", ""))

        if technology != "python":
            continue

        if not source.endswith(TARGET_SOURCE):
            continue

        text = chunk.get("text", "")
        text_lower = text.lower()

        if any(term in text_lower for term in ["generator", "generators", "yield"]):
            matches.append(chunk)

    for index, chunk in enumerate(matches, start=1):
        print()
        print(f"CHUNK {chunk.get('id')}")
        print("-" * 100)

        text = clean_text(chunk.get("text", ""))

        # Highlight why this chunk matched.
        evidence_terms = [
            term for term in ["generator", "generators", "yield"]
            if term in text.lower()
        ]

        print(f"Evidence terms: {', '.join(evidence_terms)}")
        print()
        print(text[:1200])
        print("-" * 100)

    print()
    print(f"MATCHING CHUNKS: {len(matches)}")

    if not matches:
        print("[WARNING] No generator/yield evidence found.")
    else:
        print("[OK] Generator/yield evidence found in tutorial/classes.txt")


if __name__ == "__main__":
    main()