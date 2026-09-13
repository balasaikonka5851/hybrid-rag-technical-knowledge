"""
HybridRAG - Document Chunking Pipeline

This module converts normalized documents from the ingestion
pipeline into retrieval-friendly chunks.

Supported source formats:
    - Markdown (.md)
    - Plain text (.txt)

Main responsibilities:
    1. Preserve document metadata.
    2. Detect Markdown sections.
    3. Split large sections into manageable chunks.
    4. Preserve section context.
    5. Use controlled overlap.
    6. Validate the generated chunks.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List


# =========================================================
# Configuration
# =========================================================

MAX_CHUNK_CHARS = 1200

# Characters repeated between neighboring chunks.
OVERLAP_CHARS = 200

# Chunks smaller than this are considered too small to be
# useful as independent retrieval units.
MIN_CHUNK_CHARS = 80

# When searching for a natural boundary, don't search too
# far away from the target chunk size.
BOUNDARY_SEARCH_CHARS = 400


# =========================================================
# Text cleaning
# =========================================================

def clean_chunk(text: str) -> str:
    """
    Clean text while preserving meaningful line structure.

    We intentionally do NOT remove all newlines because
    technical documentation often relies on formatting,
    lists, and code blocks.
    """

    if not isinstance(text, str):
        return ""

    # Remove null characters.
    text = text.replace("\x00", "")

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize tabs.
    text = text.replace("\t", " ")

    # Remove trailing spaces from lines.
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Prevent excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# Natural boundary detection
# =========================================================

def find_best_boundary(
    text: str,
    target_position: int
) -> int:
    """
    Find a natural place to end a chunk.

    Preference:

        1. Paragraph boundary
        2. Line boundary
        3. Sentence boundary
        4. Target position

    This avoids cutting documentation unnecessarily.
    """

    search_start = max(
        0,
        target_position - BOUNDARY_SEARCH_CHARS
    )

    candidate = text[
        search_start:target_position
    ]

    # -----------------------------------------------------
    # 1. Paragraph boundary
    # -----------------------------------------------------

    paragraph_position = candidate.rfind(
        "\n\n"
    )

    if paragraph_position != -1:

        return (
            search_start
            + paragraph_position
            + 2
        )

    # -----------------------------------------------------
    # 2. Line boundary
    # -----------------------------------------------------

    line_position = candidate.rfind(
        "\n"
    )

    if line_position != -1:

        return (
            search_start
            + line_position
            + 1
        )

    # -----------------------------------------------------
    # 3. Sentence boundary
    # -----------------------------------------------------

    sentence_matches = list(
        re.finditer(
            r"[.!?](?:\s|$)",
            candidate
        )
    )

    if sentence_matches:

        match = sentence_matches[-1]

        return (
            search_start
            + match.end()
        )

    # -----------------------------------------------------
    # 4. Last resort
    # -----------------------------------------------------

    return target_position


# =========================================================
# Split large text
# =========================================================

def split_large_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS
) -> List[str]:
    """
    Split a large block of text into overlapping chunks.

    The function tries to preserve natural boundaries rather
    than blindly cutting every N characters.
    """

    text = clean_chunk(text)

    if not text:
        return []

    if max_chars <= 0:
        raise ValueError(
            "max_chars must be greater than zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= max_chars:
        raise ValueError(
            "overlap must be smaller than max_chars."
        )

    # Small text does not need splitting.
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []

    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(
            start + max_chars,
            text_length
        )

        # -------------------------------------------------
        # Final chunk
        # -------------------------------------------------

        if target_end >= text_length:

            final_chunk = text[start:].strip()

            if final_chunk:
                chunks.append(final_chunk)

            break

        # -------------------------------------------------
        # Find natural boundary
        # -------------------------------------------------

        end = find_best_boundary(
            text,
            target_end
        )

        # Safety check.
        if end <= start:
            end = target_end

        chunk = text[
            start:end
        ].strip()

        if chunk:
            chunks.append(chunk)

        # -------------------------------------------------
        # Calculate next starting point
        # -------------------------------------------------

        next_start = end - overlap

        # Never allow infinite loops.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# =========================================================
# Markdown heading detection
# =========================================================

MARKDOWN_HEADING_PATTERN = re.compile(
    r"(?m)^(#{1,6})\s+(.+?)\s*$"
)


# =========================================================
# Markdown section extraction
# =========================================================

def extract_markdown_sections(
    text: str
) -> List[Dict]:
    """
    Extract Markdown sections based on headings.

    Example:

        # Authentication

        Some text...

        ## OAuth2

        More text...

    becomes:

        [
            {
                "section": "Authentication",
                "level": 1,
                "text": "Some text..."
            },
            {
                "section": "OAuth2",
                "level": 2,
                "text": "More text..."
            }
        ]
    """

    text = clean_chunk(text)

    if not text:
        return []

    matches = list(
        MARKDOWN_HEADING_PATTERN.finditer(text)
    )

    # -----------------------------------------------------
    # No Markdown headings
    # -----------------------------------------------------

    if not matches:

        return [
            {
                "section": "General",
                "level": 0,
                "text": text
            }
        ]

    sections: List[Dict] = []

    # -----------------------------------------------------
    # Process each heading
    # -----------------------------------------------------

    for index, match in enumerate(matches):

        level = len(
            match.group(1)
        )

        heading = match.group(2).strip()

        content_start = match.end()

        # Find the next heading.
        if index + 1 < len(matches):

            content_end = matches[
                index + 1
            ].start()

        else:

            content_end = len(text)

        content = text[
            content_start:content_end
        ].strip()

        sections.append(
            {
                "section": heading,
                "level": level,
                "text": content
            }
        )

    # -----------------------------------------------------
    # Handle text before the first heading
    # -----------------------------------------------------

    first_heading_start = matches[0].start()

    preamble = text[
        :first_heading_start
    ].strip()

    if preamble:

        sections.insert(
            0,
            {
                "section": "Preamble",
                "level": 0,
                "text": preamble
            }
        )

    return sections


# =========================================================
# Merge small pieces
# =========================================================

def merge_small_chunks(
    chunks: List[Dict]
) -> List[Dict]:
    """
    Prevent very small chunks from becoming independent
    retrieval units.

    A tiny chunk is merged into its neighboring chunk.

    This is especially useful for documentation sections
    containing only a few characters or a short sentence.
    """

    if not chunks:
        return []

    result: List[Dict] = []

    for current in chunks:

        current_text = current["text"].strip()

        # -------------------------------------------------
        # Normal chunk
        # -------------------------------------------------

        if len(current_text) >= MIN_CHUNK_CHARS:

            result.append(current)

            continue

        # -------------------------------------------------
        # Tiny chunk
        # -------------------------------------------------

        # If there is a previous chunk, merge into it.
        if result:

            previous = result[-1]

            merged_text = (
                previous["text"].rstrip()
                + "\n\n"
                + current_text
            ).strip()

            previous["text"] = merged_text

            # Record merged section information.
            existing_section = previous[
                "metadata"
            ].get("section", "")

            current_section = current[
                "metadata"
            ].get("section", "")

            if (
                current_section
                and current_section
                != existing_section
            ):

                previous[
                    "metadata"
                ]["merged_sections"] = (
                    previous[
                        "metadata"
                    ].get(
                        "merged_sections",
                        []
                    )
                    + [current_section]
                )

            continue

        # -------------------------------------------------
        # Tiny first chunk
        # -------------------------------------------------

        # Keep it temporarily. If there is no previous
        # chunk, we cannot safely discard source content.
        result.append(current)

    return result


# =========================================================
# Chunk one document
# =========================================================

def chunk_document(
    document: Dict
) -> List[Dict]:
    """
    Convert one normalized document into retrieval chunks.
    """

    if not isinstance(document, dict):

        raise TypeError(
            "document must be a dictionary."
        )

    if "text" not in document:

        raise ValueError(
            "Document is missing 'text'."
        )

    if "metadata" not in document:

        raise ValueError(
            "Document is missing 'metadata'."
        )

    text = document["text"]

    metadata = document["metadata"]

    document_id = document.get(
        "id",
        "unknown-document"
    )

    technology = metadata.get(
        "technology",
        "unknown"
    )

    source = metadata.get(
        "source",
        "unknown"
    )

    file_type = metadata.get(
        "file_type",
        ""
    ).lower()

    if not text.strip():

        return []

    raw_chunks: List[Dict] = []

    # =====================================================
    # Markdown documents
    # =====================================================

    if file_type == ".md":

        sections = extract_markdown_sections(
            text
        )

        for section_index, section in enumerate(
            sections
        ):

            section_name = section[
                "section"
            ]

            section_level = section[
                "level"
            ]

            section_text = section[
                "text"
            ]

            # -------------------------------------------------
            # IMPORTANT:
            # Include the heading in the actual chunk.
            #
            # This gives the embedding model context such as:
            #
            # "Dependencies
            #  FastAPI provides..."
            #
            # instead of only:
            #
            # "FastAPI provides..."
            # -------------------------------------------------

            if section_name != "General":

                contextual_text = (
                    f"Section: {section_name}\n\n"
                    f"{section_text}"
                )

            else:

                contextual_text = section_text

            pieces = split_large_text(
                contextual_text
            )

            for piece_index, piece in enumerate(
                pieces
            ):

                chunk_id = (
                    f"{document_id}_"
                    f"{section_index}_"
                    f"{piece_index}"
                )

                raw_chunks.append(
                    {
                        "id": chunk_id,

                        "text": piece,

                        "metadata": {
                            "document_id": document_id,
                            "technology": technology,
                            "source": source,
                            "file_name": metadata.get(
                                "file_name",
                                Path(source).name
                            ),
                            "file_type": file_type,
                            "section": section_name,
                            "section_level": section_level,
                            "chunk_index": piece_index
                        }
                    }
                )

    # =====================================================
    # Plain text documents
    # =====================================================

    elif file_type == ".txt":

        pieces = split_large_text(
            text
        )

        for piece_index, piece in enumerate(
            pieces
        ):

            chunk_id = (
                f"{document_id}_"
                f"{piece_index}"
            )

            raw_chunks.append(
                {
                    "id": chunk_id,

                    "text": piece,

                    "metadata": {
                        "document_id": document_id,
                        "technology": technology,
                        "source": source,
                        "file_name": metadata.get(
                            "file_name",
                            Path(source).name
                        ),
                        "file_type": file_type,
                        "section": "General",
                        "section_level": 0,
                        "chunk_index": piece_index
                    }
                }
            )

    # =====================================================
    # Unsupported type
    # =====================================================

    else:

        return []

    # -----------------------------------------------------
    # Merge tiny chunks
    # -----------------------------------------------------

    return merge_small_chunks(
        raw_chunks
    )


# =========================================================
# Chunk all documents
# =========================================================

def chunk_documents(
    documents: List[Dict]
) -> List[Dict]:
    """
    Chunk every loaded document.
    """

    if not isinstance(documents, list):

        raise TypeError(
            "documents must be a list."
        )

    all_chunks: List[Dict] = []

    for document_index, document in enumerate(
        documents,
        start=1
    ):

        try:

            chunks = chunk_document(
                document
            )

            all_chunks.extend(chunks)

        except Exception as error:

            source = (
                document
                .get("metadata", {})
                .get("source", "unknown")
            )

            print(
                f"[WARNING] Failed to chunk document "
                f"{document_index}: {source}"
            )

            print(
                f"          Reason: {error}"
            )

    return all_chunks


# =========================================================
# Chunk validation
# =========================================================

def validate_chunks(
    chunks: List[Dict]
) -> Dict:
    """
    Validate the generated chunk collection.

    Returns a dictionary containing validation statistics.
    """

    if not isinstance(chunks, list):

        raise TypeError(
            "chunks must be a list."
        )

    total_chunks = len(chunks)

    empty_chunks = []

    tiny_chunks = []

    duplicate_ids = []

    missing_metadata = []

    seen_ids = set()

    sizes = []

    for chunk in chunks:

        chunk_id = chunk.get(
            "id"
        )

        text = chunk.get(
            "text",
            ""
        )

        metadata = chunk.get(
            "metadata"
        )

        # -------------------------------------------------
        # ID validation
        # -------------------------------------------------

        if not chunk_id:

            missing_metadata.append(
                "missing chunk ID"
            )

        elif chunk_id in seen_ids:

            duplicate_ids.append(
                chunk_id
            )

        else:

            seen_ids.add(
                chunk_id
            )

        # -------------------------------------------------
        # Text validation
        # -------------------------------------------------

        if not text.strip():

            empty_chunks.append(
                chunk_id
            )

        else:

            size = len(text)

            sizes.append(size)

            if size < MIN_CHUNK_CHARS:

                tiny_chunks.append(
                    chunk_id
                )

        # -------------------------------------------------
        # Metadata validation
        # -------------------------------------------------

        if not isinstance(
            metadata,
            dict
        ):

            missing_metadata.append(
                f"{chunk_id}: metadata missing"
            )

            continue

        required_fields = [
            "document_id",
            "technology",
            "source",
            "file_type",
            "section"
        ]

        for field in required_fields:

            if field not in metadata:

                missing_metadata.append(
                    f"{chunk_id}: missing {field}"
                )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    if sizes:

        smallest = min(sizes)

        largest = max(sizes)

        average = (
            sum(sizes)
            / len(sizes)
        )

    else:

        smallest = 0
        largest = 0
        average = 0

    return {
        "total_chunks": total_chunks,
        "empty_chunks": len(empty_chunks),
        "tiny_chunks": len(tiny_chunks),
        "duplicate_ids": len(duplicate_ids),
        "missing_metadata": len(
            missing_metadata
        ),
        "smallest_chunk": smallest,
        "largest_chunk": largest,
        "average_chunk": average
    }


# =========================================================
# Print statistics
# =========================================================

def print_chunk_statistics(
    chunks: List[Dict]
) -> None:
    """
    Print human-readable chunk statistics.
    """

    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY")
    print("=" * 70)

    if not chunks:

        print(
            "Total chunks: 0"
        )

        print(
            "[ERROR] No chunks were created."
        )

        print("=" * 70)

        return

    validation = validate_chunks(
        chunks
    )

    print(
        f"Total chunks: "
        f"{validation['total_chunks']:,}"
    )

    print(
        f"Smallest chunk: "
        f"{validation['smallest_chunk']:,} chars"
    )

    print(
        f"Largest chunk: "
        f"{validation['largest_chunk']:,} chars"
    )

    print(
        f"Average chunk: "
        f"{validation['average_chunk']:,.0f} chars"
    )

    print(
        f"Tiny chunks (< {MIN_CHUNK_CHARS}): "
        f"{validation['tiny_chunks']:,}"
    )

    print(
        f"Empty chunks: "
        f"{validation['empty_chunks']:,}"
    )

    print(
        f"Duplicate IDs: "
        f"{validation['duplicate_ids']:,}"
    )

    print(
        f"Metadata problems: "
        f"{validation['missing_metadata']:,}"
    )

    # -----------------------------------------------------
    # Technology statistics
    # -----------------------------------------------------

    technology_counts: Dict[str, int] = {}

    for chunk in chunks:

        technology = chunk[
            "metadata"
        ].get(
            "technology",
            "unknown"
        )

        technology_counts[
            technology
        ] = (
            technology_counts.get(
                technology,
                0
            )
            + 1
        )

    print("\nChunks by technology:")

    for technology, count in sorted(
        technology_counts.items()
    ):

        print(
            f"  {technology}: {count:,}"
        )

    # -----------------------------------------------------
    # File type statistics
    # -----------------------------------------------------

    file_type_counts: Dict[str, int] = {}

    for chunk in chunks:

        file_type = chunk[
            "metadata"
        ].get(
            "file_type",
            "unknown"
        )

        file_type_counts[
            file_type
        ] = (
            file_type_counts.get(
                file_type,
                0
            )
            + 1
        )

    print("\nChunks by file type:")

    for file_type, count in sorted(
        file_type_counts.items()
    ):

        print(
            f"  {file_type}: {count:,}"
        )

    # -----------------------------------------------------
    # Validation status
    # -----------------------------------------------------

    print("\nValidation:")

    if validation["empty_chunks"] == 0:

        print(
            "  [OK] No empty chunks."
        )

    else:

        print(
            "  [ERROR] Empty chunks detected."
        )

    if validation["duplicate_ids"] == 0:

        print(
            "  [OK] All chunk IDs are unique."
        )

    else:

        print(
            "  [ERROR] Duplicate chunk IDs detected."
        )

    if validation["missing_metadata"] == 0:

        print(
            "  [OK] Required metadata is present."
        )

    else:

        print(
            "  [ERROR] Metadata problems detected."
        )

    if validation["tiny_chunks"] == 0:

        print(
            "  [OK] No tiny chunks."
        )

    else:

        print(
            "  [WARNING] Tiny chunks detected."
        )

    print("=" * 70)


# =========================================================
# Display sample chunks
# =========================================================

def display_sample_chunks(
    chunks: List[Dict],
    max_samples: int = 2
) -> None:
    """
    Display representative chunks.

    Shows one or two examples from each technology.
    """

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS")
    print("=" * 70)

    technology_samples: Dict[
        str,
        List[Dict]
    ] = {}

    for chunk in chunks:

        technology = chunk[
            "metadata"
        ].get(
            "technology",
            "unknown"
        )

        technology_samples.setdefault(
            technology,
            []
        )

        if len(
            technology_samples[
                technology
            ]
        ) < max_samples:

            technology_samples[
                technology
            ].append(chunk)

    for technology in sorted(
        technology_samples
    ):

        for chunk in technology_samples[
            technology
        ]:

            metadata = chunk[
                "metadata"
            ]

            print(
                "\n" + "-" * 70
            )

            print(
                f"Technology : "
                f"{technology}"
            )

            print(
                f"Chunk ID   : "
                f"{chunk['id']}"
            )

            print(
                f"Source     : "
                f"{metadata.get('source', 'unknown')}"
            )

            print(
                f"Section    : "
                f"{metadata.get('section', 'unknown')}"
            )

            print(
                f"Chunk size : "
                f"{len(chunk['text']):,} characters"
            )

            print(
                "\nContent preview:"
            )

            print(
                chunk["text"][:700]
            )


# =========================================================
# Main test
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Determine project root
    # -----------------------------------------------------

    PROJECT_ROOT = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # Make the project root available for imports.
    if str(PROJECT_ROOT) not in sys.path:

        sys.path.insert(
            0,
            str(PROJECT_ROOT)
        )

    # -----------------------------------------------------
    # Import document loader
    # -----------------------------------------------------

    try:

        from src.ingestion.document_loader import (
            load_documents
        )

    except ImportError as error:

        print(
            "[ERROR] Could not import document loader."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "\nExpected project structure:"
        )

        print(
            "HybridRAG/"
        )

        print(
            "├── data/raw/"
        )

        print(
            "├── src/"
        )

        print(
            "│   ├── ingestion/"
        )

        print(
            "│   │   └── document_loader.py"
        )

        print(
            "│   └── chunking/"
        )

        print(
            "│       └── chunker.py"
        )

        sys.exit(1)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    print("=" * 70)

    print(
        "HYBRID RAG - CHUNKING TEST"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # Step 1: Load documents
    # -----------------------------------------------------

    print(
        "\n[1/3] Loading documents..."
    )

    try:

        documents = load_documents()

    except Exception as error:

        print(
            "\n[ERROR] Document loading failed."
        )

        print(
            f"Reason: {error}"
        )

        sys.exit(1)

    if not documents:

        print(
            "\n[ERROR] No documents were loaded."
        )

        print(
            "Check:"
        )

        print(
            "  data/raw/python/"
        )

        print(
            "  data/raw/fastapi/"
        )

        sys.exit(1)

    print(
        f"\nSuccessfully loaded "
        f"{len(documents):,} documents."
    )

    # -----------------------------------------------------
    # Step 2: Create chunks
    # -----------------------------------------------------

    print(
        "\n[2/3] Creating chunks..."
    )

    try:

        chunks = chunk_documents(
            documents
        )

    except Exception as error:

        print(
            "\n[ERROR] Chunking failed."
        )

        print(
            f"Reason: {error}"
        )

        sys.exit(1)

    if not chunks:

        print(
            "\n[ERROR] No chunks were created."
        )

        sys.exit(1)

    print(
        f"Successfully created "
        f"{len(chunks):,} chunks."
    )

    # -----------------------------------------------------
    # Step 3: Validate
    # -----------------------------------------------------

    print(
        "\n[3/3] Validating chunks..."
    )

    print_chunk_statistics(
        chunks
    )

    # -----------------------------------------------------
    # Samples
    # -----------------------------------------------------

    display_sample_chunks(
        chunks,
        max_samples=2
    )

    # -----------------------------------------------------
    # Final status
    # -----------------------------------------------------

    validation = validate_chunks(
        chunks
    )

    has_critical_error = (
        validation["empty_chunks"] > 0
        or validation["duplicate_ids"] > 0
        or validation["missing_metadata"] > 0
    )

    print(
        "\n" + "=" * 70
    )

    if has_critical_error:

        print(
            "CHUNKING COMPLETED WITH ERRORS"
        )

        print(
            "Fix the validation problems "
            "before continuing."
        )

        print(
            "=" * 70
        )

        sys.exit(1)

    print(
        "CHUNKING TEST COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)