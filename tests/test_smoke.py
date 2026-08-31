import subprocess
import sys

import pytest


def test_version_import():
    from rag_mcp import __version__
    assert __version__ == "0.1.0"


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "rag_mcp", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "rag-mcp 0.1.0" in result.stdout


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "rag_mcp", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "index" in result.stdout
    assert "stats" in result.stdout


def test_store_defaults():
    from rag_mcp.store import DEFAULT_MODEL, DEFAULT_DATA_DIR
    assert "multilingual" in DEFAULT_MODEL
    assert "rag-mcp" in str(DEFAULT_DATA_DIR)


def test_ingest_supported_formats():
    from rag_mcp.ingest import SUPPORTED
    assert ".md" in SUPPORTED
    assert ".pdf" in SUPPORTED
    assert ".txt" in SUPPORTED


def test_chunk_markdown():
    from rag_mcp.ingest import chunk_markdown
    text = (
        "# Title\n\n"
        "This is the first paragraph with enough content to pass the length filter.\n\n"
        "## Section\n\n"
        "This is the second paragraph with enough content to pass the length filter."
    )
    chunks = chunk_markdown(text, max_size=1000, overlap=0)
    assert len(chunks) >= 1
    assert any("first paragraph" in c for c in chunks)


def test_parse_markdown_frontmatter():
    import tempfile
    from pathlib import Path
    from rag_mcp.ingest import parse_file

    content = "---\ntitle: Test\ndate: 2025-01-01\n---\n\n# Hello\n\nBody text."
    with tempfile.NamedTemporaryFile(suffix=".md", mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        text, meta = parse_file(Path(f.name))

    assert "Body text" in text
    assert meta.get("title") == "Test"
