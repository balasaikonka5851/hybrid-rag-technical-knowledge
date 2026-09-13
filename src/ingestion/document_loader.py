from pathlib import Path
import hashlib
import re
from typing import List, Dict


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_DIR = Path("data/raw")

SUPPORTED_EXTENSIONS = {".txt", ".md"}

MIN_DOCUMENT_CHARS = 50


# ---------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean raw documentation text while preserving useful structure.
    """

    if not isinstance(text, str):
        return ""

    # Remove null characters
    text = text.replace("\x00", "")

    # Normalize Windows line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces
    text = text.replace("\t", " ")

    # Remove trailing spaces from each line
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Reduce excessive spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


# ---------------------------------------------------------
# Document ID
# ---------------------------------------------------------

def create_document_id(
    technology: str,
    source: str
) -> str:
    """
    Create a deterministic ID for a document.
    """

    raw_id = f"{technology}:{source}"

    return hashlib.sha1(
        raw_id.encode("utf-8")
    ).hexdigest()


# ---------------------------------------------------------
# Load one file
# ---------------------------------------------------------

def load_single_file(
    file_path: Path,
    technology: str
) -> Dict | None:

    try:

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    except Exception as error:

        print(
            f"[WARNING] Could not read: {file_path}"
        )

        print(f"          Reason: {error}")

        return None

    text = clean_text(text)

    # Ignore empty/tiny files
    if len(text) < MIN_DOCUMENT_CHARS:

        print(
            f"[SKIPPED] Too little content: {file_path}"
        )

        return None

    relative_source = file_path.as_posix()

    document_id = create_document_id(
        technology,
        relative_source
    )

    return {
        "id": document_id,

        "text": text,

        "metadata": {
            "technology": technology,
            "source": relative_source,
            "file_name": file_path.name,
            "file_type": file_path.suffix.lower(),
            "character_count": len(text)
        }
    }


# ---------------------------------------------------------
# Load all documents
# ---------------------------------------------------------

def load_documents(
    data_dir: Path = DATA_DIR
) -> List[Dict]:

    documents = []

    if not data_dir.exists():

        raise FileNotFoundError(
            f"Data directory does not exist: {data_dir}"
        )

    # Sort for deterministic results
    technology_directories = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name.lower()
    )

    for technology_dir in technology_directories:

        technology = technology_dir.name

        files = sorted(
            [
                path
                for path in technology_dir.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )
            ],
            key=lambda path: str(path).lower()
        )

        print(
            f"\nLoading {technology}: "
            f"{len(files)} supported files found"
        )

        for file_path in files:

            document = load_single_file(
                file_path,
                technology
            )

            if document is not None:
                documents.append(document)

    return documents


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

def print_statistics(documents: List[Dict]):

    print("\n" + "=" * 60)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 60)

    print(f"Total documents: {len(documents)}")

    technology_counts = {}
    file_type_counts = {}

    total_characters = 0

    for document in documents:

        metadata = document["metadata"]

        technology = metadata["technology"]
        file_type = metadata["file_type"]

        technology_counts[technology] = (
            technology_counts.get(
                technology,
                0
            ) + 1
        )

        file_type_counts[file_type] = (
            file_type_counts.get(
                file_type,
                0
            ) + 1
        )

        total_characters += metadata[
            "character_count"
        ]

    print("\nDocuments by technology:")

    for technology, count in sorted(
        technology_counts.items()
    ):
        print(
            f"  {technology}: {count}"
        )

    print("\nDocuments by file type:")

    for file_type, count in sorted(
        file_type_counts.items()
    ):
        print(
            f"  {file_type}: {count}"
        )

    print(
        f"\nTotal characters: {total_characters:,}"
    )

    if documents:

        average = (
            total_characters
            / len(documents)
        )

        print(
            f"Average document size: "
            f"{average:,.0f} characters"
        )

    print("=" * 60)


# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print(
        f"Scanning documentation directory: "
        f"{DATA_DIR.resolve()}"
    )

    documents = load_documents()

    print_statistics(documents)

    print("\nSample documents:")

    for document in documents[:5]:

        metadata = document["metadata"]

        print("\n" + "-" * 60)

        print("ID:", document["id"])

        print(
            "Technology:",
            metadata["technology"]
        )

        print(
            "Source:",
            metadata["source"]
        )

        print(
            "Type:",
            metadata["file_type"]
        )

        print(
            "Characters:",
            metadata["character_count"]
        )

        print("\nPreview:")
        print(
            document["text"][:300]
        )