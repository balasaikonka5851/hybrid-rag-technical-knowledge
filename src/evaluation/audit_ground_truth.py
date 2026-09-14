from pathlib import Path
import json
import re
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"
QUERIES_PATH = ROOT / "data" / "evaluation" / "queries.json"

AUDIT_QUERY_IDS = {
    "q03","q06","q09","q12","q13","q14","q17","q21",
    "q23","q24","q31","q32","q35","q36","q37","q38"
}

def normalize_source(value):
    s = str(value).strip().replace("\\", "/")
    s = re.sub(r"^[A-Za-z]:/", "", s)
    s = re.sub(r"^/+", "", s)
    prefixes = ("data/raw/","data/processed/","raw/","processed/","fastapi/","python/")
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if s.lower().startswith(p):
                s = s[len(p):]
                changed = True
    return s.lower().strip("/")

def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_chunks(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("chunks","data","documents","records"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Could not find a chunk list in chunks.json.")

def metadata(chunk):
    m = chunk.get("metadata", {})
    return m if isinstance(m, dict) else {}

def main():
    print("=" * 100)
    print("HYBRID RAG - GROUND TRUTH AUDIT")
    print("=" * 100)

    chunks = get_chunks(load_json(CHUNKS_PATH))
    queries = load_json(QUERIES_PATH)

    source_chunks = defaultdict(list)

    for chunk in chunks:
        m = metadata(chunk)
        source = normalize_source(m.get("source", chunk.get("source", "")))
        tech = str(m.get("technology", chunk.get("technology", ""))).lower()
        if source:
            source_chunks[(tech, source)].append(chunk)

    print(f"\n[OK] Corpus chunks: {len(chunks):,}")
    print(f"[OK] Evaluation queries: {len(queries)}")

    missing = []
    matched = 0

    for q in queries:
        tech = str(q.get("technology", "")).lower()
        for source in q.get("relevant_sources", []):
            key = (tech, normalize_source(source))
            if key in source_chunks:
                matched += 1
            else:
                missing.append((q.get("id"), tech, normalize_source(source)))

    print(f"[OK] Matched source declarations: {matched}")
    print(f"[OK] Missing source declarations: {len(missing)}")

    if missing:
        print("\nMissing declarations:")
        for item in missing:
            print("  ", item)

    print("\n" + "=" * 100)
    print("MANUAL AUDIT OF WEAK / FAILED QUERIES")
    print("=" * 100)

    for q in queries:
        qid = q.get("id")
        if qid not in AUDIT_QUERY_IDS:
            continue

        tech = str(q.get("technology", "")).lower()
        print(f"\n{qid} [{tech}] {q.get('query','')}")
        print("Ground truth:")

        for source in q.get("relevant_sources", []):
            norm = normalize_source(source)
            matches = source_chunks.get((tech, norm), [])
            print(f"  - {norm}  | chunks={len(matches)}")

            if not matches:
                filename = Path(norm).name
                close = []
                for (t, s), vals in source_chunks.items():
                    if t == tech and Path(s).name == filename:
                        close.append((s, len(vals)))
                if close:
                    print("    Close filename matches:")
                    for s, count in sorted(close):
                        print(f"      {s} ({count} chunks)")
                continue

            for i, chunk in enumerate(matches[:2], 1):
                text = re.sub(r"\s+", " ", str(chunk.get("text", "")).strip())
                print(f"    chunk {i}: {text[:500]}")

    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

if __name__ == "__main__":
    main()
