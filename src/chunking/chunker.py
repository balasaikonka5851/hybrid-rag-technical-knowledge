"""
HybridRAG - Robust Document Chunking Pipeline

Converts normalized documents from the ingestion pipeline
into validated, retrieval-friendly chunks.

Supported formats:
    - Markdown (.md)
    - Plain text (.txt)

Pipeline:

    Documents
        ↓
    Clean text
        ↓
    Markdown section detection
        ↓
    Natural-boundary splitting
        ↓
    Tiny-chunk merging
        ↓
    Validation
        ↓
    chunks.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

CHUNKS_FILE = (
    PROCESSED_DIR
    / "chunks.json"
)


# ============================================================
# CHUNK CONFIGURATION
# ============================================================

MAX_CHUNK_CHARS = 1200

OVERLAP_CHARS = 200

MIN_CHUNK_CHARS = 64

BOUNDARY_SEARCH_CHARS = 400


# ============================================================
# MARKDOWN HEADING
# ============================================================

MARKDOWN_HEADING_PATTERN = re.compile(
    r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$"
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean text while preserving technical formatting.
    """

    if not isinstance(text, str):
        return ""

    # Remove null characters.
    text = text.replace(
        "\x00",
        ""
    )

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )

    # Normalize tabs.
    text = text.replace(
        "\t",
        " "
    )

    # Remove trailing whitespace.
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Collapse excessive blank lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# NATURAL BOUNDARY
# ============================================================

def find_best_boundary(
    text: str,
    target_position: int
) -> int:
    """
    Find the best natural boundary before target_position.

    Priority:

        paragraph
        line
        sentence
        word
        exact position
    """

    search_start = max(
        0,
        target_position
        - BOUNDARY_SEARCH_CHARS
    )

    candidate = text[
        search_start:
        target_position
    ]

    # --------------------------------------------------------
    # Paragraph
    # --------------------------------------------------------

    paragraph_position = candidate.rfind(
        "\n\n"
    )

    if paragraph_position != -1:

        return (
            search_start
            + paragraph_position
            + 2
        )

    # --------------------------------------------------------
    # Line
    # --------------------------------------------------------

    line_position = candidate.rfind(
        "\n"
    )

    if line_position != -1:

        return (
            search_start
            + line_position
            + 1
        )

    # --------------------------------------------------------
    # Sentence
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Word
    # --------------------------------------------------------

    whitespace_matches = list(
        re.finditer(
            r"\s+",
            candidate
        )
    )

    if whitespace_matches:

        match = whitespace_matches[-1]

        return (
            search_start
            + match.end()
        )

    return target_position


# ============================================================
# SPLIT TEXT
# ============================================================

def split_large_text(
    text: str,
    max_chars: int = MAX_CHUNK_CHARS,
    overlap: int = OVERLAP_CHARS
) -> List[str]:
    """
    Split text into overlapping retrieval chunks.

    Important:
    A final fragment smaller than MIN_CHUNK_CHARS is merged
    with the previous chunk instead of being returned alone.
    """

    text = clean_text(
        text
    )

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

    # --------------------------------------------------------
    # Small text
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Final chunk
        # ----------------------------------------------------

        if target_end >= text_length:

            final_chunk = text[
                start:
            ].strip()

            if not final_chunk:
                break

            # If final fragment is tiny, merge it into the
            # previous chunk rather than returning it alone.
            if (
                len(final_chunk)
                < MIN_CHUNK_CHARS
                and chunks
            ):

                merged = (
                    chunks[-1].rstrip()
                    + "\n\n"
                    + final_chunk.lstrip()
                ).strip()

                chunks[-1] = merged

            else:

                chunks.append(
                    final_chunk
                )

            break

        # ----------------------------------------------------
        # Natural boundary
        # ----------------------------------------------------

        end = find_best_boundary(
            text,
            target_end
        )

        if end <= start:

            end = target_end

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        # ----------------------------------------------------
        # Overlap
        # ----------------------------------------------------

        next_start = end - overlap

        if next_start <= start:

            next_start = end

        start = next_start

    return chunks


# ============================================================
# MARKDOWN SECTION EXTRACTION
# ============================================================

def extract_markdown_sections(
    text: str
) -> List[Dict[str, Any]]:
    """
    Extract Markdown sections.

    Text before the first heading becomes "Preamble".
    """

    text = clean_text(
        text
    )

    if not text:

        return []

    matches = list(
        MARKDOWN_HEADING_PATTERN.finditer(
            text
        )
    )

    # --------------------------------------------------------
    # No headings
    # --------------------------------------------------------

    if not matches:

        return [
            {
                "section": "General",
                "level": 0,
                "text": text
            }
        ]

    sections: List[
        Dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Preamble
    # --------------------------------------------------------

    preamble = text[
        :matches[0].start()
    ].strip()

    if preamble:

        sections.append(
            {
                "section": "Preamble",
                "level": 0,
                "text": preamble
            }
        )

    # --------------------------------------------------------
    # Headings
    # --------------------------------------------------------

    for index, match in enumerate(
        matches
    ):

        level = len(
            match.group(1)
        )

        heading = (
            match
            .group(2)
            .strip()
        )

        content_start = match.end()

        if index + 1 < len(matches):

            content_end = matches[
                index + 1
            ].start()

        else:

            content_end = len(text)

        content = text[
            content_start:
            content_end
        ].strip()

        if not content:
            continue

        sections.append(
            {
                "section": heading,
                "level": level,
                "text": content
            }
        )

    return sections


# ============================================================
# CREATE CHUNK
# ============================================================

def create_chunk(
    chunk_id: str,
    text: str,
    document_id: str,
    technology: str,
    source: str,
    file_name: str,
    file_type: str,
    section: str,
    section_level: int,
    chunk_index: int
) -> Dict[str, Any]:
    """
    Create a consistent chunk object.
    """

    return {
        "id": chunk_id,

        "text": clean_text(
            text
        ),

        "metadata": {
            "document_id": document_id,
            "technology": technology,
            "source": source,
            "file_name": file_name,
            "file_type": file_type,
            "section": section,
            "section_level": section_level,
            "chunk_index": chunk_index
        }
    }


# ============================================================
# MERGE TINY CHUNKS
# ============================================================

def merge_tiny_chunks(
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Robustly merge tiny chunks.

    Handles:

        normal → tiny
        tiny → normal
        tiny → tiny → normal
        tiny → tiny → tiny
        tiny at document end

    We never silently discard source text.
    """

    if not chunks:

        return []

    result: List[
        Dict[str, Any]
    ] = []

    tiny_buffer: List[
        Dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # First pass
    # --------------------------------------------------------

    for chunk in chunks:

        text = clean_text(
            chunk.get(
                "text",
                ""
            )
        )

        if not text:
            continue

        chunk["text"] = text

        # Normal chunk.
        if len(text) >= MIN_CHUNK_CHARS:

            # Attach all preceding tiny chunks.
            if tiny_buffer:

                prefix_parts = [
                    item["text"]
                    for item in tiny_buffer
                ]

                prefix = "\n\n".join(
                    prefix_parts
                )

                chunk["text"] = (
                    prefix
                    + "\n\n"
                    + chunk["text"]
                ).strip()

                merged_ids = chunk[
                    "metadata"
                ].setdefault(
                    "merged_chunk_ids",
                    []
                )

                for item in tiny_buffer:

                    merged_ids.append(
                        item["id"]
                    )

                tiny_buffer.clear()

            result.append(
                chunk
            )

        # Tiny chunk.
        else:

            tiny_buffer.append(
                chunk
            )

    # --------------------------------------------------------
    # Remaining tiny chunks
    # --------------------------------------------------------

    if tiny_buffer:

        tiny_text = "\n\n".join(
            item["text"]
            for item in tiny_buffer
        )

        # Best option: merge with previous chunk.
        if result:

            previous = result[-1]

            previous["text"] = (
                previous["text"].rstrip()
                + "\n\n"
                + tiny_text.lstrip()
            ).strip()

            merged_ids = previous[
                "metadata"
            ].setdefault(
                "merged_chunk_ids",
                []
            )

            for item in tiny_buffer:

                merged_ids.append(
                    item["id"]
                )

        else:

            # This means the entire section/document consists
            # only of tiny content.
            #
            # Preserve it rather than silently deleting it.
            combined = tiny_buffer[0].copy()

            combined["text"] = (
                tiny_text
            )

            result.append(
                combined
            )

    return result


# ============================================================
# CHUNK ONE DOCUMENT
# ============================================================

def chunk_document(
    document: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Convert one normalized document into chunks.
    """

    if not isinstance(
        document,
        dict
    ):

        raise TypeError(
            "document must be a dictionary."
        )

    text = document.get(
        "text",
        ""
    )

    metadata = document.get(
        "metadata",
        {}
    )

    if not isinstance(
        text,
        str
    ):

        raise ValueError(
            "document.text must be a string."
        )

    if not isinstance(
        metadata,
        dict
    ):

        raise ValueError(
            "document.metadata must be a dictionary."
        )

    text = clean_text(
        text
    )

    if not text:

        return []

    document_id = str(
        document.get(
            "id",
            "unknown-document"
        )
    )

    technology = str(
        metadata.get(
            "technology",
            "unknown"
        )
    )

    source = str(
        metadata.get(
            "source",
            "unknown"
        )
    )

    file_type = str(
        metadata.get(
            "file_type",
            ""
        )
    ).lower()

    file_name = str(
        metadata.get(
            "file_name"
        )
        or Path(source).name
        or "unknown"
    )

    raw_chunks: List[
        Dict[str, Any]
    ] = []

    # ========================================================
    # MARKDOWN
    # ========================================================

    if file_type == ".md":

        sections = extract_markdown_sections(
            text
        )

        for section_index, section in enumerate(
            sections
        ):

            section_name = str(
                section["section"]
            )

            section_level = int(
                section["level"]
            )

            section_text = str(
                section["text"]
            )

            # Add section context.
            if section_name not in {
                "General",
                "Preamble"
            }:

                contextual_text = (
                    f"Section: "
                    f"{section_name}\n\n"
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
                    create_chunk(
                        chunk_id=chunk_id,
                        text=piece,
                        document_id=document_id,
                        technology=technology,
                        source=source,
                        file_name=file_name,
                        file_type=file_type,
                        section=section_name,
                        section_level=section_level,
                        chunk_index=piece_index
                    )
                )

    # ========================================================
    # PLAIN TEXT
    # ========================================================

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
                create_chunk(
                    chunk_id=chunk_id,
                    text=piece,
                    document_id=document_id,
                    technology=technology,
                    source=source,
                    file_name=file_name,
                    file_type=file_type,
                    section="General",
                    section_level=0,
                    chunk_index=piece_index
                )
            )

    else:

        return []

    # --------------------------------------------------------
    # Robust tiny-chunk cleanup
    # --------------------------------------------------------

    cleaned_chunks = merge_tiny_chunks(
        raw_chunks
    )

    return cleaned_chunks


# ============================================================
# CHUNK ALL DOCUMENTS
# ============================================================

def chunk_documents(
    documents: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Chunk every document.

    Individual document failures are reported without
    crashing the complete process.
    """

    if not isinstance(
        documents,
        list
    ):

        raise TypeError(
            "documents must be a list."
        )

    all_chunks: List[
        Dict[str, Any]
    ] = []

    failed_documents = 0

    for index, document in enumerate(
        documents,
        start=1
    ):

        try:

            chunks = chunk_document(
                document
            )

            all_chunks.extend(
                chunks
            )

        except Exception as error:

            failed_documents += 1

            source = (
                document
                .get(
                    "metadata",
                    {}
                )
                .get(
                    "source",
                    "unknown"
                )
                if isinstance(
                    document,
                    dict
                )
                else "unknown"
            )

            print(
                f"[WARNING] Failed document "
                f"{index}: {source}"
            )

            print(
                f"          Reason: {error}"
            )

    if failed_documents:

        print(
            f"\n[WARNING] Failed documents: "
            f"{failed_documents}"
        )

    return all_chunks


# ============================================================
# VALIDATION
# ============================================================

def validate_chunks(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate all chunks.
    """

    if not isinstance(
        chunks,
        list
    ):

        raise TypeError(
            "chunks must be a list."
        )

    seen_ids = set()

    duplicate_ids = []

    empty_chunks = []

    tiny_chunks = []

    metadata_problems = []

    oversized_chunks = []

    sizes = []

    required_fields = [
        "document_id",
        "technology",
        "source",
        "file_name",
        "file_type",
        "section",
        "section_level",
        "chunk_index"
    ]

    for position, chunk in enumerate(
        chunks
    ):

        if not isinstance(
            chunk,
            dict
        ):

            metadata_problems.append(
                f"position {position}: "
                f"invalid chunk object"
            )

            continue

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

        # ----------------------------------------------------
        # ID
        # ----------------------------------------------------

        if not chunk_id:

            metadata_problems.append(
                f"position {position}: "
                f"missing ID"
            )

        elif chunk_id in seen_ids:

            duplicate_ids.append(
                str(chunk_id)
            )

        else:

            seen_ids.add(
                chunk_id
            )

        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if (
            not isinstance(
                text,
                str
            )
            or not text.strip()
        ):

            empty_chunks.append(
                str(chunk_id)
            )

        else:

            size = len(text)

            sizes.append(
                size
            )

            if size < MIN_CHUNK_CHARS:

                tiny_chunks.append(
                    str(chunk_id)
                )

            if size > (
                MAX_CHUNK_CHARS
                + MIN_CHUNK_CHARS
            ):

                oversized_chunks.append(
                    str(chunk_id)
                )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        if not isinstance(
            metadata,
            dict
        ):

            metadata_problems.append(
                f"{chunk_id}: "
                f"metadata missing"
            )

            continue

        for field in required_fields:

            value = metadata.get(
                field
            )

            if value is None or value == "":

                metadata_problems.append(
                    f"{chunk_id}: "
                    f"missing {field}"
                )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    smallest = (
        min(sizes)
        if sizes
        else 0
    )

    largest = (
        max(sizes)
        if sizes
        else 0
    )

    average = (
        sum(sizes) / len(sizes)
        if sizes
        else 0
    )

    return {
        "total_chunks": len(chunks),
        "empty_chunks": len(empty_chunks),
        "tiny_chunks": len(tiny_chunks),
        "duplicate_ids": len(duplicate_ids),
        "metadata_problems": len(
            metadata_problems
        ),
        "oversized_chunks": len(
            oversized_chunks
        ),
        "smallest_chunk": smallest,
        "largest_chunk": largest,
        "average_chunk": average
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_chunk_statistics(
    chunks: List[Dict[str, Any]]
) -> None:

    validation = validate_chunks(
        chunks
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHUNKING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total chunks        : "
        f"{validation['total_chunks']:,}"
    )

    print(
        f"Smallest chunk      : "
        f"{validation['smallest_chunk']:,} chars"
    )

    print(
        f"Largest chunk       : "
        f"{validation['largest_chunk']:,} chars"
    )

    print(
        f"Average chunk       : "
        f"{validation['average_chunk']:,.0f} chars"
    )

    print(
        f"Tiny chunks (< {MIN_CHUNK_CHARS}) : "
        f"{validation['tiny_chunks']:,}"
    )

    print(
        f"Empty chunks        : "
        f"{validation['empty_chunks']:,}"
    )

    print(
        f"Duplicate IDs       : "
        f"{validation['duplicate_ids']:,}"
    )

    print(
        f"Metadata problems   : "
        f"{validation['metadata_problems']:,}"
    )

    print(
        f"Oversized chunks    : "
        f"{validation['oversized_chunks']:,}"
    )

    # ========================================================
    # TECHNOLOGY
    # ========================================================

    technology_counts: Dict[
        str,
        int
    ] = {}

    file_type_counts: Dict[
        str,
        int
    ] = {}

    for chunk in chunks:

        metadata = chunk[
            "metadata"
        ]

        technology = metadata.get(
            "technology",
            "unknown"
        )

        file_type = metadata.get(
            "file_type",
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

        file_type_counts[
            file_type
        ] = (
            file_type_counts.get(
                file_type,
                0
            )
            + 1
        )

    print(
        "\nChunks by technology:"
    )

    for technology, count in sorted(
        technology_counts.items()
    ):

        print(
            f"  {technology:<12}: "
            f"{count:,}"
        )

    print(
        "\nChunks by file type:"
    )

    for file_type, count in sorted(
        file_type_counts.items()
    ):

        print(
            f"  {file_type:<12}: "
            f"{count:,}"
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\nValidation:"
    )

    if validation[
        "empty_chunks"
    ] == 0:

        print(
            "  [OK] No empty chunks."
        )

    else:

        print(
            "  [ERROR] Empty chunks detected."
        )

    if validation[
        "duplicate_ids"
    ] == 0:

        print(
            "  [OK] All chunk IDs are unique."
        )

    else:

        print(
            "  [ERROR] Duplicate chunk IDs detected."
        )

    if validation[
        "metadata_problems"
    ] == 0:

        print(
            "  [OK] Required metadata is present."
        )

    else:

        print(
            "  [ERROR] Metadata problems detected."
        )

    if validation[
        "tiny_chunks"
    ] == 0:

        print(
            "  [OK] No tiny chunks."
        )

    else:

        print(
            "  [ERROR] Tiny chunks detected."
        )

    if validation[
        "oversized_chunks"
    ] == 0:

        print(
            "  [OK] No unexpectedly oversized chunks."
        )

    else:

        print(
            "  [WARNING] Oversized chunks detected."
        )

    print(
        "=" * 70
    )


# ============================================================
# SAMPLE CHUNKS
# ============================================================

def display_sample_chunks(
    chunks: List[Dict[str, Any]],
    max_samples_per_technology: int = 2
) -> None:

    samples: Dict[
        str,
        List[Dict[str, Any]]
    ] = {}

    for chunk in chunks:

        technology = chunk[
            "metadata"
        ].get(
            "technology",
            "unknown"
        )

        samples.setdefault(
            technology,
            []
        )

        if len(
            samples[technology]
        ) < max_samples_per_technology:

            samples[
                technology
            ].append(
                chunk
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SAMPLE CHUNKS"
    )

    print(
        "=" * 70
    )

    for technology in sorted(
        samples
    ):

        for chunk in samples[
            technology
        ]:

            metadata = chunk[
                "metadata"
            ]

            print(
                "\n"
                + "-" * 70
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


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(
    chunks: List[Dict[str, Any]]
) -> None:
    """
    Atomically save chunks.json and verify it.
    """

    if not chunks:

        raise ValueError(
            "Cannot save zero chunks."
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = (
        PROCESSED_DIR
        / "chunks.json.tmp"
    )

    try:

        # ----------------------------------------------------
        # Write temporary file
        # ----------------------------------------------------

        with temporary_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2
            )

            file.flush()

        # ----------------------------------------------------
        # Replace final file
        # ----------------------------------------------------

        temporary_file.replace(
            CHUNKS_FILE
        )

    finally:

        if temporary_file.exists():

            temporary_file.unlink(
                missing_ok=True
            )

    # --------------------------------------------------------
    # Verify file
    # --------------------------------------------------------

    if not CHUNKS_FILE.exists():

        raise RuntimeError(
            "chunks.json was not created."
        )

    file_size = (
        CHUNKS_FILE.stat().st_size
    )

    if file_size <= 0:

        raise RuntimeError(
            "chunks.json is empty."
        )

    # Read back.
    with CHUNKS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        saved_chunks = json.load(
            file
        )

    if not isinstance(
        saved_chunks,
        list
    ):

        raise RuntimeError(
            "chunks.json does not contain a JSON list."
        )

    if len(
        saved_chunks
    ) != len(chunks):

        raise RuntimeError(
            "Saved chunk count does not match "
            "generated chunk count."
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHUNK FILE SAVED"
    )

    print(
        "=" * 70
    )

    print(
        f"File      : "
        f"{CHUNKS_FILE}"
    )

    print(
        f"Chunks    : "
        f"{len(chunks):,}"
    )

    print(
        f"File size : "
        f"{file_size / (1024 * 1024):.2f} MB"
    )

    print(
        "[OK] chunks.json created and verified."
    )

    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "HYBRID RAG - CHUNKING PIPELINE"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # IMPORT
    # ========================================================

    try:

        from src.ingestion.document_loader import (
            load_documents
        )

    except ImportError as error:

        print(
            "\n[ERROR] Could not import document loader."
        )

        print(
            f"Reason: {error}"
        )

        raise SystemExit(1)

    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/4] Loading documents..."
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

        raise SystemExit(1)

    if not documents:

        print(
            "\n[ERROR] No documents were loaded."
        )

        raise SystemExit(1)

    print(
        f"\nSuccessfully loaded "
        f"{len(documents):,} documents."
    )

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/4] Creating chunks..."
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

        raise SystemExit(1)

    if not chunks:

        print(
            "\n[ERROR] No chunks were created."
        )

        raise SystemExit(1)

    print(
        f"Successfully created "
        f"{len(chunks):,} chunks."
    )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/4] Validating chunks..."
    )

    print_chunk_statistics(
        chunks
    )

    validation = validate_chunks(
        chunks
    )

    critical_errors = (
        validation[
            "empty_chunks"
        ] > 0

        or validation[
            "duplicate_ids"
        ] > 0

        or validation[
            "metadata_problems"
        ] > 0

        or validation[
            "tiny_chunks"
        ] > 0
    )

    if critical_errors:

        print(
            "\n[ERROR] Chunk validation failed."
        )

        print(
            "chunks.json will NOT be written."
        )

        raise SystemExit(1)

    display_sample_chunks(
        chunks
    )

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/4] Saving chunks..."
    )

    try:

        save_chunks(
            chunks
        )

    except Exception as error:

        print(
            "\n[ERROR] Failed to save chunks."
        )

        print(
            f"Reason: {error}"
        )

        raise SystemExit(1)

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CHUNKING PIPELINE COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        f"Documents loaded : "
        f"{len(documents):,}"
    )

    print(
        f"Chunks created   : "
        f"{len(chunks):,}"
    )

    print(
        f"Output file      : "
        f"{CHUNKS_FILE}"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if str(PROJECT_ROOT) not in sys.path:

        sys.path.insert(
            0,
            str(PROJECT_ROOT)
        )

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n\n[STOPPED] Pipeline interrupted."
        )

        sys.exit(1)

    except SystemExit:

        raise

    except Exception as error:

        print(
            "\n\n[FATAL ERROR]"
        )

        print(
            f"{error}"
        )

        sys.exit(1)