"""MCP server exposing multilingual RAG search tools."""
from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from rag_mcp.ingest import ingest_directory, parse_file
from rag_mcp.store import VectorStore

mcp = FastMCP("rag-mcp")

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def _docs_dirs() -> list[Path]:
    raw = os.environ.get("RAG_DOCS", "")
    return [Path(p.strip()).expanduser().resolve() for p in raw.split(",") if p.strip()]


@mcp.tool()
def search(query: str, n: int = 5) -> str:
    """Search the knowledge base with a natural-language query.

    Works across languages: query in English to find Russian content,
    or query in Russian to find English content. The multilingual
    embedding model maps semantically similar concepts across languages.
    """
    store = _get_store()
    results = store.search(query, n)
    if not results:
        return "No results found. If the index is empty, call the reindex tool first."

    parts = []
    for r in results:
        meta = r["metadata"]
        score = max(0, 1 - r["distance"])
        parts.append(
            f"### {meta.get('title', '?')} (score: {score:.2f})\n"
            f"**Source:** {meta.get('source', '?')}  \n"
            f"**Category:** {meta.get('category', '?')}\n\n"
            f"{r['text']}\n"
        )
    return "\n---\n".join(parts)


@mcp.tool()
def get_document(path: str) -> str:
    """Retrieve the full content of a source document by path.

    Accepts an absolute path or a path relative to any configured RAG_DOCS directory.
    """
    p = Path(path)
    if not p.exists():
        for d in _docs_dirs():
            candidate = d / path
            if candidate.exists():
                p = candidate
                break
        else:
            return f"Not found: {path}"

    text, meta = parse_file(p)
    return f"# {meta.get('title', p.name)}\n\n{text}"


@mcp.tool()
def list_documents(filter: str | None = None) -> str:
    """List indexed documents. Optionally filter by keyword in title or path."""
    docs = _get_store().list_documents()
    if filter:
        fl = filter.lower()
        docs = [
            d
            for d in docs
            if fl in d["title"].lower() or fl in d["source"].lower()
        ]

    if not docs:
        return "No documents indexed. Call reindex to populate the knowledge base."

    lines = ["| Title | Category | Chunks |", "|---|---|---|"]
    for d in docs:
        lines.append(f"| {d['title'][:60]} | {d['category'][:20]} | {d['chunks']} |")
    lines.append(f"\n**Total: {len(docs)} documents**")
    return "\n".join(lines)


@mcp.tool()
def reindex() -> str:
    """Re-index all documents from the configured RAG_DOCS directories."""
    dirs = _docs_dirs()
    if not dirs:
        return (
            "No RAG_DOCS configured. Set the RAG_DOCS environment variable to a "
            "comma-separated list of directories containing your documents."
        )

    store = _get_store()
    chunk_size = int(os.environ.get("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap = int(os.environ.get("RAG_CHUNK_OVERLAP", "200"))

    results = []
    for d in dirs:
        if not d.is_dir():
            results.append(f"{d}: not a directory, skipped")
            continue
        s = ingest_directory(store, d, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        results.append(
            f"{d.name}: {s['indexed']} docs, {s['chunks']} chunks, "
            f"{s['skipped']} skipped, {len(s['errors'])} errors"
        )
        for err in s["errors"][:5]:
            results.append(f"  ERROR: {err}")

    total = store.stats()
    results.append(
        f"\nTotal: {total['total_documents']} documents, {total['total_chunks']} chunks"
    )
    return "\n".join(results)


@mcp.tool()
def stats() -> str:
    """Get knowledge base statistics: document count, chunk count, categories."""
    return json.dumps(_get_store().stats(), indent=2, ensure_ascii=False)
