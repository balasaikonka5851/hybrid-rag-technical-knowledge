"""
HybridRAG - Query-Aware Retrieval Policy

Profiles technical queries and selects conservative
semantic/BM25 fusion weights.

Design goals:
- Robust query normalization
- Accurate code/API identifier detection
- Conservative adaptive weighting
- Avoid treating ordinary technical words as identifiers
- Explicit query classification
- Safe handling of short and long queries
- Deterministic behavior
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any


# ============================================================
# Query profile
# ============================================================

@dataclass(frozen=True)
class QueryProfile:
    normalized_query: str
    token_count: int
    technical_token_count: int

    has_code_identifier: bool
    has_question_word: bool
    has_error_term: bool
    has_api_term: bool
    has_how_to_pattern: bool

    query_type: str

    semantic_weight: float
    bm25_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================
# Tokenization
# ============================================================

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    """
    Tokenize query while preserving underscores.

    Example:
        response_model -> response_model
        RequestValidationError -> RequestValidationError
    """
    return _TOKEN_RE.findall(text.lower())


# ============================================================
# Identifier detection
# ============================================================

# snake_case / underscored identifiers
_SNAKE_CASE_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+\b"
)

# dotted identifiers such as:
#   app.include_router
#   pydantic.BaseModel
#   fastapi.FastAPI
_DOTTED_IDENTIFIER_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b"
)

# Well-known class/API naming conventions.
#
# Examples:
#   HTTPException
#   RequestValidationError
#   JSONResponse
#   BaseModel
#   APIRouter
#   FastAPI
#   CORSMiddleware
#
# We intentionally require CamelCase / PascalCase structure
# rather than treating every word ending in "Error" as an
# identifier.
_PASCAL_IDENTIFIER_RE = re.compile(
    r"\b"
    r"(?=[A-Z][A-Za-z0-9]*[a-z])"
    r"[A-Z][A-Za-z0-9]*"
    r"(?:Error|Exception|Request|Response|Model|Config|"
    r"Router|Middleware|Dependency)"
    r"\b"
)

# Generic identifier forms that are strongly suggestive of code.
#
# Examples:
#   HTTPException
#   RequestValidationError
#   response_model
#   include_router
#
# Do NOT use ordinary words such as "exceptions" or "errors".
_IDENTIFIER_PATTERNS = (
    _SNAKE_CASE_RE,
    _DOTTED_IDENTIFIER_RE,
    _PASCAL_IDENTIFIER_RE,
)


def _has_code_identifier(query: str) -> bool:
    """
    Detect explicit code/API identifiers.

    Important:
    Ordinary technical vocabulary such as:
        exception
        exceptions
        error
        dependencies
        validation

    must NOT automatically become identifiers.
    """
    return any(pattern.search(query) for pattern in _IDENTIFIER_PATTERNS)


# ============================================================
# Technical token detection
# ============================================================

def _is_technical_token(token: str) -> bool:
    """
    Determine whether a token looks like a code identifier.

    The token is already lowercased, so PascalCase information
    is unavailable here. Therefore underscore identifiers are
    the strongest token-level signal.
    """
    if "_" in token:
        return True

    return False


# ============================================================
# Question vocabulary
# ============================================================

_QUESTION_WORDS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "which",
    "can",
    "does",
    "do",
    "is",
    "are",
    "should",
    "could",
    "would",
}


# ============================================================
# API vocabulary
# ============================================================

_API_TERMS = {
    "api",
    "endpoint",
    "endpoints",
    "route",
    "routes",
    "router",
    "routers",
    "request",
    "requests",
    "response",
    "responses",
    "dependency",
    "dependencies",
    "middleware",
    "middlewares",
    "parameter",
    "parameters",
    "status",
    "validation",
    "authentication",
    "authorization",
    "fastapi",
    "http",
    "https",
    "pydantic",
    "schema",
    "schemas",
    "query",
    "path",
    "header",
    "headers",
    "cookie",
    "cookies",
    "body",
    "websocket",
    "websockets",
}


# ============================================================
# Error vocabulary
# ============================================================

_ERROR_TERMS = {
    "error",
    "errors",
    "exception",
    "exceptions",
    "traceback",
    "failure",
    "failures",
    "validationerror",
    "exceptionhandler",
    "httpException".lower(),
}


# ============================================================
# How-to detection
# ============================================================

_HOW_TO_RE = re.compile(
    r"""
    \b
    how
    \s+
    (?:
        do
        |
        does
        |
        can
        |
        should
        |
        to
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# Additional imperative patterns.

_HOW_TO_IMPERATIVE_RE = re.compile(
    r"""
    \b
    (?:
        create
        |
        build
        |
        use
        |
        configure
        |
        implement
        |
        add
        |
        define
        |
        handle
        |
        install
        |
        integrate
        |
        connect
        |
        deploy
        |
        call
        |
        run
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# Query normalization
# ============================================================

def _normalize_query(query: str) -> str:
    """
    Normalize whitespace without destroying identifiers.
    """
    return " ".join(query.strip().split())


# ============================================================
# Main profiler
# ============================================================

def profile_query(query: str) -> QueryProfile:
    """
    Profile a query and choose retrieval fusion weights.

    Weight philosophy:

    General:
        0.75 semantic / 0.25 BM25

    Conceptual:
        0.80 semantic / 0.20 BM25

    Technical identifier:
        0.65 semantic / 0.35 BM25

    API identifier:
        0.60 semantic / 0.40 BM25

    How-to:
        semantic-heavy, because conceptual similarity and
        procedural language are usually important.

    The classifier is intentionally conservative.
    """

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    if not isinstance(query, str):
        raise TypeError("query must be a string")

    normalized = _normalize_query(query)

    if not normalized:
        raise ValueError("query cannot be empty")

    # Protect the retrieval system from pathological queries.
    if len(normalized) > 2000:
        raise ValueError(
            "query is too long; maximum supported length is 2000 characters"
        )

    lowered = normalized.lower()

    # --------------------------------------------------------
    # Token features
    # --------------------------------------------------------

    tokens = _tokens(normalized)

    technical_tokens = [
        token
        for token in tokens
        if _is_technical_token(token)
    ]

    has_identifier = _has_code_identifier(normalized)

    has_error = (
        any(term in tokens for term in _ERROR_TERMS)
        or bool(
            re.search(
                r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b",
                normalized,
            )
        )
    )

    has_api = any(
        term in tokens
        for term in _API_TERMS
    )

    has_question = bool(
        tokens
        and tokens[0] in _QUESTION_WORDS
    )

    has_how_to = bool(
        _HOW_TO_RE.search(normalized)
        or _HOW_TO_IMPERATIVE_RE.search(normalized)
    )

    # --------------------------------------------------------
    # Conservative baseline
    # --------------------------------------------------------

    semantic = 0.75
    bm25 = 0.25
    query_type = "general"

    # --------------------------------------------------------
    # Classification hierarchy
    #
    # Highest priority:
    #   API + exact identifier
    #
    # Then:
    #   exact identifier
    #
    # Then:
    #   conceptual question
    #
    # Then:
    #   how-to
    #
    # Error vocabulary alone is NOT an identifier.
    # --------------------------------------------------------

    # 1. Explicit API identifier
    if has_api and has_identifier:
        semantic = 0.60
        bm25 = 0.40
        query_type = "api_identifier"

    # 2. Explicit technical/code identifier
    elif has_identifier:
        semantic = 0.65
        bm25 = 0.35
        query_type = "technical_identifier"

    # 3. Broad conceptual question
    elif has_question:
        semantic = 0.80
        bm25 = 0.20
        query_type = "conceptual"

    # 4. How-to / procedural query
    elif has_how_to:
        semantic = 0.78
        bm25 = 0.22
        query_type = "how_to"

    # 5. Technical error concept without identifier
    #
    # Example:
    #   "How does Python handle exceptions?"
    #
    # This is conceptual, NOT technical_identifier.
    elif has_error:
        semantic = 0.78
        bm25 = 0.22
        query_type = "conceptual"

    # --------------------------------------------------------
    # How-to adjustment
    #
    # How-to queries should generally remain semantic-heavy.
    # However, an explicit identifier must retain its stronger
    # lexical/BM25 contribution.
    # --------------------------------------------------------

    if has_how_to:

        if query_type == "api_identifier":
            # Keep API identifier balance.
            semantic = max(semantic, 0.60)
            bm25 = min(bm25, 0.40)

        elif query_type == "technical_identifier":
            # Keep identifier-aware retrieval.
            semantic = max(semantic, 0.65)
            bm25 = min(bm25, 0.35)

        else:
            # Pure procedural/conceptual query.
            semantic = max(semantic, 0.78)
            bm25 = min(bm25, 0.22)

            if query_type == "general":
                query_type = "how_to"

    # --------------------------------------------------------
    # Short-query safety
    #
    # Very short queries benefit more from lexical matching
    # when they contain an explicit identifier.
    # --------------------------------------------------------

    if len(tokens) <= 2 and has_identifier:

        if has_api:
            semantic = 0.55
            bm25 = 0.45
            query_type = "api_identifier"
        else:
            semantic = 0.60
            bm25 = 0.40
            query_type = "technical_identifier"

    # --------------------------------------------------------
    # Long broad conceptual queries
    #
    # Longer natural-language questions generally contain
    # enough semantic information to favor embeddings.
    # --------------------------------------------------------

    if (
        len(tokens) >= 10
        and not has_identifier
        and not has_error
    ):
        semantic = max(semantic, 0.82)
        bm25 = min(bm25, 0.18)

        if query_type == "general":
            query_type = "conceptual"

    # --------------------------------------------------------
    # Final safety normalization
    # --------------------------------------------------------

    semantic = max(0.0, float(semantic))
    bm25 = max(0.0, float(bm25))

    total = semantic + bm25

    if total <= 0:
        semantic = 0.75
        bm25 = 0.25
        total = 1.0

    semantic /= total
    bm25 /= total

    # --------------------------------------------------------
    # Return immutable profile
    # --------------------------------------------------------

    return QueryProfile(
        normalized_query=normalized,
        token_count=len(tokens),
        technical_token_count=len(technical_tokens),
        has_code_identifier=has_identifier,
        has_question_word=has_question,
        has_error_term=has_error,
        has_api_term=has_api,
        has_how_to_pattern=has_how_to,
        query_type=query_type,
        semantic_weight=round(semantic, 4),
        bm25_weight=round(bm25, 4),
    )


# ============================================================
# CLI / manual diagnostic
# ============================================================

if __name__ == "__main__":

    test_queries = [
        "How does Python handle exceptions?",
        "What is RequestValidationError in FastAPI?",
        "How do I create dependencies in FastAPI?",
        "How do I use response_model in FastAPI?",
        "What is response_model?",
        "How does app.include_router work?",
        "What is HTTPException?",
        "What is semantic search?",
        "Explain Python decorators",
        "How can I configure middleware?",
        "What is BaseModel?",
        "How does dependency injection work?",
    ]

    print("=" * 80)
    print("HybridRAG Query Profiling Diagnostic")
    print("=" * 80)

    for query in test_queries:

        profile = profile_query(query)

        print()
        print(f"Query: {query}")
        print(f"Type: {profile.query_type}")
        print(
            f"Weights: "
            f"semantic={profile.semantic_weight:.2f}, "
            f"bm25={profile.bm25_weight:.2f}"
        )
        print(
            f"Identifier={profile.has_code_identifier}, "
            f"API={profile.has_api_term}, "
            f"Error={profile.has_error_term}, "
            f"HowTo={profile.has_how_to_pattern}"
        )

    print()
    print("=" * 80)