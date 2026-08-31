"""Multilingual RAG MCP server — cross-lingual search over local documents."""

__version__ = "0.2.0"

from multilingual_rag_mcp.store import VectorStore
from multilingual_rag_mcp.ingest import ingest_directory, parse_file, chunk_markdown

__all__ = ["VectorStore", "ingest_directory", "parse_file", "chunk_markdown"]
