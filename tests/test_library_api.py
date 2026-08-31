"""Tests for the public library API: imports, named collections, and metadata filtering."""
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_data_dir(tmp_path):
    return tmp_path / "rag_data"


def test_public_imports():
    from multilingual_rag_mcp import VectorStore, ingest_directory, parse_file, chunk_markdown
    assert callable(VectorStore)
    assert callable(ingest_directory)
    assert callable(parse_file)
    assert callable(chunk_markdown)


def test_named_collection(tmp_data_dir):
    from multilingual_rag_mcp import VectorStore

    store = VectorStore(data_dir=tmp_data_dir, collection="february")
    store.add_chunks("doc1", ["hello world test content here"], [{"doc_id": "doc1"}])
    assert store._col.name == "february"
    assert store._col.count() == 1

    store2 = VectorStore(data_dir=tmp_data_dir, collection="march")
    assert store2._col.count() == 0


def test_default_collection(tmp_data_dir):
    from multilingual_rag_mcp import VectorStore

    store = VectorStore(data_dir=tmp_data_dir)
    assert store._col.name == "documents"


def test_search_with_where_filter(tmp_data_dir):
    from multilingual_rag_mcp import VectorStore

    store = VectorStore(data_dir=tmp_data_dir, collection="test_filter")
    store.add_chunks(
        "notes_doc",
        ["Protein synthesis and amino acids in cellular biology"],
        [{"doc_id": "notes_doc", "source_type": "notes", "lesson": "biology"}],
    )
    store.add_chunks(
        "transcript_doc",
        ["Protein folding and molecular structure discussion"],
        [{"doc_id": "transcript_doc", "source_type": "transcript", "lesson": "biology"}],
    )
    store.add_chunks(
        "slide_doc",
        ["Carbohydrate metabolism overview slide content"],
        [{"doc_id": "slide_doc", "source_type": "slide_ocr", "lesson": "chemistry"}],
    )

    results_notes = store.search("protein", n=5, where={"source_type": "notes"})
    assert len(results_notes) == 1
    assert results_notes[0]["metadata"]["source_type"] == "notes"

    results_transcript = store.search("protein", n=5, where={"source_type": "transcript"})
    assert len(results_transcript) == 1
    assert results_transcript[0]["metadata"]["source_type"] == "transcript"

    results_lesson = store.search("biology", n=5, where={"lesson": "biology"})
    assert len(results_lesson) == 2


def test_search_without_where(tmp_data_dir):
    from multilingual_rag_mcp import VectorStore

    store = VectorStore(data_dir=tmp_data_dir, collection="test_no_filter")
    store.add_chunks("d1", ["alpha content here"], [{"doc_id": "d1", "source_type": "notes"}])
    store.add_chunks("d2", ["beta content here"], [{"doc_id": "d2", "source_type": "transcript"}])

    results = store.search("content", n=5)
    assert len(results) == 2


def test_search_empty_collection(tmp_data_dir):
    from multilingual_rag_mcp import VectorStore

    store = VectorStore(data_dir=tmp_data_dir, collection="empty")
    assert store.search("anything", n=5, where={"source_type": "notes"}) == []


def test_ingest_directory_with_named_collection(tmp_data_dir, tmp_path):
    from multilingual_rag_mcp import VectorStore, ingest_directory

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "test.md").write_text(
        "# Test Document\n\n"
        "This is a test document with enough content to pass the minimum length filter "
        "for chunking and indexing into the vector store.\n\n"
        "## Section Two\n\n"
        "More content here to ensure we have sufficient text for the chunking algorithm "
        "to produce at least one chunk from this markdown file."
    )

    store = VectorStore(data_dir=tmp_data_dir, collection="my_collection")
    stats = ingest_directory(store, docs_dir, category="test_docs")

    assert stats["indexed"] == 1
    assert stats["chunks"] >= 1
    assert store._col.name == "my_collection"

    results = store.search("test document")
    assert len(results) >= 1
