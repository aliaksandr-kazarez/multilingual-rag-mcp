"""Document parsing, chunking, and indexing."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SUPPORTED = {".md", ".txt", ".pdf", ".json", ".csv"}


def parse_file(path: Path) -> tuple[str, dict]:
    """Parse a file → (text, metadata). Strips markdown frontmatter into metadata."""
    meta: dict = {"source": str(path), "filename": path.name}
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        import pymupdf

        doc = pymupdf.open(str(path))
        text = "\n\n".join(page.get_text() for page in doc)
        meta["title"] = doc.metadata.get("title") or path.stem
        doc.close()
        return text, meta

    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".json":
        try:
            data = json.loads(raw)
            text = json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            text = raw
        meta["title"] = path.stem
        return text, meta

    if suffix == ".md":
        text, fm = _strip_frontmatter(raw)
        meta.update(fm)
        if "title" not in meta:
            meta["title"] = _first_heading(text) or path.stem
    else:
        text = raw
        meta["title"] = path.stem

    return text, meta


def _strip_frontmatter(text: str) -> tuple[str, dict]:
    """Remove YAML frontmatter and return (body, parsed fields)."""
    if not text.startswith("---"):
        return text, {}
    end = text.find("\n---", 3)
    if end < 0:
        return text, {}
    fm_block = text[3:end]
    body = text[end + 4 :].strip()
    fields: dict = {}
    for line in fm_block.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            fields[key.strip()] = val.strip().strip("\"'")
    return body, fields


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def chunk_markdown(text: str, max_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split markdown into chunks, respecting headers and code blocks."""
    sections = re.split(r"(?=^#{1,3} )", text, flags=re.MULTILINE)
    chunks: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= max_size:
            if len(section) > 30:
                chunks.append(section)
        else:
            chunks.extend(_split_paragraphs(section, max_size, overlap))
    return chunks


def _split_paragraphs(
    text: str, max_size: int = 1000, overlap: int = 200
) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 > max_size and current:
            chunks.append(current.strip())
            tail = current[-overlap:].strip() if overlap else ""
            current = (tail + "\n\n" + para) if tail else para
        else:
            current = (current + "\n\n" + para) if current else para
    if current.strip() and len(current.strip()) > 30:
        chunks.append(current.strip())
    return chunks


def doc_id_from_path(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def ingest_directory(
    store,
    docs_dir: Path,
    *,
    category: str = "",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """Index all supported files under docs_dir."""
    stats = {"indexed": 0, "chunks": 0, "skipped": 0, "errors": []}

    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED:
            continue
        if any(part.startswith(".") for part in path.relative_to(docs_dir).parts):
            continue

        try:
            text, meta = parse_file(path)
            if not text or len(text.strip()) < 50:
                stats["skipped"] += 1
                continue

            did = doc_id_from_path(path, docs_dir)
            meta["doc_id"] = did
            meta["category"] = category or docs_dir.name

            chunks = chunk_markdown(text, chunk_size, chunk_overlap)
            if not chunks:
                stats["skipped"] += 1
                continue

            metadatas = [{**meta, "chunk_index": i} for i in range(len(chunks))]
            store.add_chunks(did, chunks, metadatas)
            stats["indexed"] += 1
            stats["chunks"] += len(chunks)
        except Exception as e:
            stats["errors"].append(f"{path.name}: {e}")

    return stats


def index_cli(args: list[str]) -> None:
    """CLI entry: `rag-mcp index /path/to/docs [/more/docs ...]`"""
    import os

    dirs = args or [
        p.strip()
        for p in os.environ.get("RAG_DOCS", "").split(",")
        if p.strip()
    ]
    if not dirs:
        print("Usage: rag-mcp index <docs_dir> [<docs_dir> ...]", file=sys.stderr)
        print("   or: RAG_DOCS=/path/to/docs rag-mcp index", file=sys.stderr)
        sys.exit(1)

    from multilingual_rag_mcp.store import VectorStore

    store = VectorStore()
    chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))

    for d in dirs:
        p = Path(d).expanduser().resolve()
        if not p.is_dir():
            print(f"SKIP {p}: not a directory", file=sys.stderr)
            continue
        print(f"Indexing {p} ...", file=sys.stderr)
        s = ingest_directory(store, p, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        print(
            f"  {s['indexed']} docs, {s['chunks']} chunks, "
            f"{s['skipped']} skipped, {len(s['errors'])} errors",
            file=sys.stderr,
        )
        for err in s["errors"][:10]:
            print(f"  ERROR: {err}", file=sys.stderr)

    total = store.stats()
    print(
        f"\nDone. {total['total_documents']} documents, "
        f"{total['total_chunks']} chunks in index.",
        file=sys.stderr,
    )
