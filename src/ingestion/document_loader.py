from pathlib import Path
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import pymupdf
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from openpyxl import load_workbook
from pptx import Presentation


# =========================================================
# Configuration
# =========================================================

DATA_DIR = Path("data/raw")

MIN_DOCUMENT_CHARS = 50

# Plain-text / source-code formats that can safely be decoded
TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".text",

    # Programming languages
    ".py",
    ".pyw",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",

    # Web
    ".html",
    ".htm",
    ".css",
    ".scss",

    # Data / configuration
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".env",

    # Database
    ".sql",

    # Shell / scripting
    ".sh",
    ".bash",
    ".ps1",
    ".bat",
    ".cmd",

    # Documentation / logs
    ".log",
    ".tex",
}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS
    | {
        ".pdf",
        ".docx",
        ".xlsx",
        ".xlsm",
        ".pptx",
        ".pptm",
        ".csv",
        ".xml",
    }
)


# =========================================================
# Text cleaning
# =========================================================

def clean_text(text: str) -> str:
    """
    Clean extracted text while preserving useful documentation structure.
    """

    if not isinstance(text, str):
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")

    # Remove trailing whitespace line-by-line.
    text = "\n".join(
        line.rstrip()
        for line in text.splitlines()
    )

    # Avoid enormous blank-line sequences.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Avoid excessive spaces without destroying indentation entirely.
    text = re.sub(r"[ ]{3,}", "  ", text)

    return text.strip()


# =========================================================
# Document ID
# =========================================================

def create_document_id(
    technology: str,
    source: str,
    extra: str = "",
) -> str:
    """
    Create a deterministic SHA-1 document ID.

    extra is useful for page/slide/sheet based documents.
    """

    raw_id = f"{technology}:{source}:{extra}"

    return hashlib.sha1(
        raw_id.encode("utf-8")
    ).hexdigest()


# =========================================================
# Metadata
# =========================================================

def build_metadata(
    *,
    file_path: Path,
    technology: str,
    data_dir: Path,
    file_type: Optional[str] = None,
    page: Optional[int] = None,
    section: Optional[str] = None,
    title: Optional[str] = None,
    sheet: Optional[str] = None,
    slide: Optional[int] = None,
    paragraph: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Build consistent metadata across all supported formats.
    """

    try:
        relative_path = file_path.relative_to(data_dir)
    except ValueError:
        relative_path = file_path

    source = relative_path.as_posix()

    metadata = {
        "technology": technology,
        "source": source,
        "relative_path": source,
        "file_name": file_path.name,
        "file_type": (
            file_type
            or file_path.suffix.lower()
        ),
        "character_count": 0,
    }

    if page is not None:
        metadata["page"] = page

    if section:
        metadata["section"] = section

    if title:
        metadata["title"] = title

    if sheet:
        metadata["sheet"] = sheet

    if slide is not None:
        metadata["slide"] = slide

    if paragraph is not None:
        metadata["paragraph"] = paragraph

    if extra:
        metadata.update(extra)

    return metadata


def make_document(
    text: str,
    metadata: Dict,
    *,
    technology: str,
    source: str,
    extra_id: str = "",
) -> Optional[Dict]:
    """
    Normalize extracted content into the project's document schema.
    """

    text = clean_text(text)

    if len(text) < MIN_DOCUMENT_CHARS:
        return None

    metadata = dict(metadata)
    metadata["character_count"] = len(text)

    document_id = create_document_id(
        technology,
        source,
        extra_id,
    )

    return {
        "id": document_id,
        "text": text,
        "metadata": metadata,
    }


# =========================================================
# Plain text / source files
# =========================================================

def load_text_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    try:
        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as error:
        print(
            f"[WARNING] Could not read: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    metadata = build_metadata(
        file_path=file_path,
        technology=technology,
        data_dir=data_dir,
    )

    document = make_document(
        text,
        metadata,
        technology=technology,
        source=metadata["source"],
    )

    return [document] if document else []


# =========================================================
# PDF
# =========================================================

def load_pdf_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    documents = []

    try:
        pdf = pymupdf.open(file_path)
    except Exception as error:
        print(
            f"[WARNING] Could not open PDF: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    try:
        pdf_title = (
            pdf.metadata.get("title")
            if pdf.metadata
            else None
        )

        for page_number, page in enumerate(
            pdf,
            start=1,
        ):
            text = page.get_text("text")

            metadata = build_metadata(
                file_path=file_path,
                technology=technology,
                data_dir=data_dir,
                page=page_number,
                title=pdf_title,
            )

            document = make_document(
                text,
                metadata,
                technology=technology,
                source=metadata["source"],
                extra_id=f"page:{page_number}",
            )

            if document:
                documents.append(document)

    finally:
        pdf.close()

    return documents


# =========================================================
# DOCX
# =========================================================

def load_docx_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    documents = []

    try:
        doc = DocxDocument(file_path)
    except Exception as error:
        print(
            f"[WARNING] Could not open DOCX: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    current_heading = None
    paragraph_number = 0

    for paragraph in doc.paragraphs:

        text = paragraph.text.strip()

        if not text:
            continue

        paragraph_number += 1

        style_name = (
            paragraph.style.name
            if paragraph.style
            else ""
        )

        if style_name.startswith("Heading"):
            current_heading = text

        metadata = build_metadata(
            file_path=file_path,
            technology=technology,
            data_dir=data_dir,
            section=current_heading,
            paragraph=paragraph_number,
        )

        document = make_document(
            text,
            metadata,
            technology=technology,
            source=metadata["source"],
            extra_id=f"paragraph:{paragraph_number}",
        )

        if document:
            documents.append(document)

    # DOCX tables
    for table_number, table in enumerate(
        doc.tables,
        start=1,
    ):

        rows = []

        for row in table.rows:
            values = [
                cell.text.strip()
                for cell in row.cells
            ]

            if any(values):
                rows.append(" | ".join(values))

        table_text = "\n".join(rows)

        metadata = build_metadata(
            file_path=file_path,
            technology=technology,
            data_dir=data_dir,
            extra={
                "content_type": "table",
                "table": table_number,
            },
        )

        document = make_document(
            table_text,
            metadata,
            technology=technology,
            source=metadata["source"],
            extra_id=f"table:{table_number}",
        )

        if document:
            documents.append(document)

    return documents


# =========================================================
# HTML
# =========================================================

def load_html_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    try:
        raw_html = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as error:
        print(
            f"[WARNING] Could not read HTML: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    soup = BeautifulSoup(
        raw_html,
        "html.parser",
    )

    # Remove content that is not useful for retrieval.
    for element in soup(
        ["script", "style", "noscript"]
    ):
        element.decompose()

    title = (
        soup.title.get_text(
            " ",
            strip=True,
        )
        if soup.title
        else None
    )

    parts = []

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code"]
    ):
        text = element.get_text(
            " ",
            strip=True,
        )

        if not text:
            continue

        tag = element.name

        if tag.startswith("h"):
            parts.append(
                f"\n{element.get_text(' ', strip=True)}\n"
            )
        else:
            parts.append(text)

    text = "\n".join(parts)

    metadata = build_metadata(
        file_path=file_path,
        technology=technology,
        data_dir=data_dir,
        title=title,
    )

    document = make_document(
        text,
        metadata,
        technology=technology,
        source=metadata["source"],
    )

    return [document] if document else []


# =========================================================
# JSON
# =========================================================

def load_json_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    try:
        raw = file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        data = json.loads(raw)

        text = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        )

    except Exception as error:
        print(
            f"[WARNING] Could not parse JSON: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    metadata = build_metadata(
        file_path=file_path,
        technology=technology,
        data_dir=data_dir,
        extra={
            "content_type": "json",
        },
    )

    document = make_document(
        text,
        metadata,
        technology=technology,
        source=metadata["source"],
    )

    return [document] if document else []


# =========================================================
# CSV
# =========================================================

def load_csv_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    try:
        with file_path.open(
            "r",
            encoding="utf-8",
            errors="ignore",
            newline="",
        ) as file:

            reader = csv.reader(file)

            rows = [
                row
                for row in reader
                if any(cell.strip() for cell in row)
            ]

    except Exception as error:
        print(
            f"[WARNING] Could not parse CSV: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    text = "\n".join(
        " | ".join(cell.strip() for cell in row)
        for row in rows
    )

    metadata = build_metadata(
        file_path=file_path,
        technology=technology,
        data_dir=data_dir,
        extra={
            "content_type": "csv",
            "row_count": len(rows),
        },
    )

    document = make_document(
        text,
        metadata,
        technology=technology,
        source=metadata["source"],
    )

    return [document] if document else []


# =========================================================
# XML
# =========================================================

def load_xml_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        text_parts = []

        for element in root.iter():
            if element.text and element.text.strip():
                text_parts.append(
                    element.text.strip()
                )

        text = "\n".join(text_parts)

    except Exception as error:
        print(
            f"[WARNING] Could not parse XML: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    metadata = build_metadata(
        file_path=file_path,
        technology=technology,
        data_dir=data_dir,
        extra={
            "content_type": "xml",
            "root_tag": root.tag,
        },
    )

    document = make_document(
        text,
        metadata,
        technology=technology,
        source=metadata["source"],
    )

    return [document] if document else []


# =========================================================
# XLSX / XLSM
# =========================================================

def load_excel_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    documents = []

    try:
        workbook = load_workbook(
            filename=file_path,
            read_only=True,
            data_only=True,
        )
    except Exception as error:
        print(
            f"[WARNING] Could not open Excel file: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    try:
        for worksheet in workbook.worksheets:

            rows = []

            for row in worksheet.iter_rows(
                values_only=True
            ):
                values = [
                    str(value).strip()
                    for value in row
                    if value is not None
                    and str(value).strip()
                ]

                if values:
                    rows.append(
                        " | ".join(values)
                    )

            text = "\n".join(rows)

            metadata = build_metadata(
                file_path=file_path,
                technology=technology,
                data_dir=data_dir,
                sheet=worksheet.title,
                extra={
                    "content_type": "spreadsheet",
                    "row_count": len(rows),
                },
            )

            document = make_document(
                text,
                metadata,
                technology=technology,
                source=metadata["source"],
                extra_id=f"sheet:{worksheet.title}",
            )

            if document:
                documents.append(document)

    finally:
        workbook.close()

    return documents


# =========================================================
# PowerPoint
# =========================================================

def load_powerpoint_file(
    file_path: Path,
    technology: str,
    data_dir: Path,
) -> List[Dict]:

    documents = []

    try:
        presentation = Presentation(file_path)
    except Exception as error:
        print(
            f"[WARNING] Could not open PowerPoint: {file_path}"
        )
        print(f"          Reason: {error}")
        return []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):

        parts = []

        for shape in slide.shapes:

            if not hasattr(shape, "text"):
                continue

            text = shape.text.strip()

            if text:
                parts.append(text)

        slide_text = "\n".join(parts)

        metadata = build_metadata(
            file_path=file_path,
            technology=technology,
            data_dir=data_dir,
            slide=slide_number,
            extra={
                "content_type": "presentation",
            },
        )

        document = make_document(
            slide_text,
            metadata,
            technology=technology,
            source=metadata["source"],
            extra_id=f"slide:{slide_number}",
        )

        if document:
            documents.append(document)

    return documents


# =========================================================
# Dispatcher
# =========================================================

def load_single_file(
    file_path: Path,
    technology: str,
    data_dir: Path = DATA_DIR,
) -> List[Dict]:
    """
    Load one file using the appropriate parser.

    Returns a list because one physical file can produce
    multiple logical documents, e.g. PDF pages or Excel sheets.
    """

    extension = file_path.suffix.lower()

    if extension in TEXT_EXTENSIONS:

        # HTML has a dedicated parser.
        if extension in {".html", ".htm"}:
            return load_html_file(
                file_path,
                technology,
                data_dir,
            )

        # JSON has a dedicated parser.
        if extension == ".json":
            return load_json_file(
                file_path,
                technology,
                data_dir,
            )

        # XML has a dedicated parser.
        if extension == ".xml":
            return load_xml_file(
                file_path,
                technology,
                data_dir,
            )

        return load_text_file(
            file_path,
            technology,
            data_dir,
        )

    if extension == ".pdf":
        return load_pdf_file(
            file_path,
            technology,
            data_dir,
        )

    if extension == ".docx":
        return load_docx_file(
            file_path,
            technology,
            data_dir,
        )

    if extension in {".xlsx", ".xlsm"}:
        return load_excel_file(
            file_path,
            technology,
            data_dir,
        )

    if extension in {".pptx", ".pptm"}:
        return load_powerpoint_file(
            file_path,
            technology,
            data_dir,
        )

    if extension == ".csv":
        return load_csv_file(
            file_path,
            technology,
            data_dir,
        )

    print(
        f"[SKIPPED] Unsupported file type: {file_path}"
    )

    return []


# =========================================================
# Load all documents
# =========================================================

def load_documents(
    data_dir: Path = DATA_DIR,
) -> List[Dict]:

    documents = []

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: "
            f"{data_dir}"
        )

    technology_directories = sorted(
        [
            path
            for path in data_dir.iterdir()
            if path.is_dir()
        ],
        key=lambda path: path.name.lower(),
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
            key=lambda path: str(path).lower(),
        )

        print(
            f"\nLoading {technology}: "
            f"{len(files)} supported files found"
        )

        for file_path in files:

            loaded_documents = load_single_file(
                file_path=file_path,
                technology=technology,
                data_dir=data_dir,
            )

            documents.extend(
                loaded_documents
            )

    return documents


# =========================================================
# Statistics
# =========================================================

def print_statistics(
    documents: List[Dict],
):

    print("\n" + "=" * 60)
    print("DOCUMENT LOADING SUMMARY")
    print("=" * 60)

    print(
        f"Total logical documents: "
        f"{len(documents)}"
    )

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
                0,
            ) + 1
        )

        file_type_counts[file_type] = (
            file_type_counts.get(
                file_type,
                0,
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
        f"\nTotal characters: "
        f"{total_characters:,}"
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


# =========================================================
# Validation
# =========================================================

def validate_documents(
    documents: List[Dict],
) -> None:

    ids = [
        document["id"]
        for document in documents
    ]

    duplicate_ids = (
        len(ids) != len(set(ids))
    )

    invalid_documents = []

    for document in documents:

        if not document.get("id"):
            invalid_documents.append(
                "missing id"
            )
            continue

        if not isinstance(
            document.get("text"),
            str,
        ):
            invalid_documents.append(
                f"{document['id']}: invalid text"
            )
            continue

        if not isinstance(
            document.get("metadata"),
            dict,
        ):
            invalid_documents.append(
                f"{document['id']}: invalid metadata"
            )

    print("\nValidation:")

    print(
        f"  Duplicate IDs: "
        f"{'YES' if duplicate_ids else 'NO'}"
    )

    print(
        f"  Invalid documents: "
        f"{len(invalid_documents)}"
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate document IDs detected."
        )

    if invalid_documents:
        raise ValueError(
            "Invalid document records detected."
        )

    print("  Status: PASS")


# =========================================================
# Test
# =========================================================

if __name__ == "__main__":

    print(
        "Scanning documentation directory:"
    )

    print(
        f"  {DATA_DIR.resolve()}"
    )

    print(
        "\nSupported extensions:"
    )

    print(
        f"  {len(SUPPORTED_EXTENSIONS)} formats"
    )

    documents = load_documents()

    validate_documents(documents)

    print_statistics(documents)

    print("\nSample logical documents:")

    for document in documents[:5]:

        metadata = document["metadata"]

        print("\n" + "-" * 60)

        print(
            "ID:",
            document["id"],
        )

        print(
            "Technology:",
            metadata["technology"],
        )

        print(
            "Source:",
            metadata["source"],
        )

        print(
            "Type:",
            metadata["file_type"],
        )

        if "page" in metadata:
            print(
                "Page:",
                metadata["page"],
            )

        if "section" in metadata:
            print(
                "Section:",
                metadata["section"],
            )

        if "sheet" in metadata:
            print(
                "Sheet:",
                metadata["sheet"],
            )

        if "slide" in metadata:
            print(
                "Slide:",
                metadata["slide"],
            )

        print(
            "Characters:",
            metadata["character_count"],
        )

        print("\nPreview:")

        print(
            document["text"][:300]
        )