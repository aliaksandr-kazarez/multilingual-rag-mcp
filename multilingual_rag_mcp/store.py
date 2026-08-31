"""Vector store backed by ChromaDB with multilingual embeddings."""
from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "rag-mcp"


class _FastEmbedEF(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def __call__(self, input: Documents) -> Embeddings:
        return [e.tolist() for e in self._model.embed(list(input))]


class VectorStore:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        model: str | None = None,
        collection: str = "documents",
    ):
        data_dir = Path(data_dir or os.environ.get("RAG_DATA", str(DEFAULT_DATA_DIR)))
        data_dir.mkdir(parents=True, exist_ok=True)
        self._model_name = model or os.environ.get("RAG_MODEL", DEFAULT_MODEL)

        self._ef = _FastEmbedEF(self._model_name)
        self._client = chromadb.PersistentClient(path=str(data_dir / "chroma"))
        self._col = self._client.get_or_create_collection(
            name=collection,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self, doc_id: str, chunks: list[str], metadatas: list[dict]
    ) -> None:
        ids = [f"{doc_id}::{i}" for i in range(len(chunks))]
        self._col.upsert(ids=ids, documents=chunks, metadatas=metadatas)

    def remove_document(self, doc_id: str) -> int:
        existing = self._col.get(where={"doc_id": doc_id})
        if existing["ids"]:
            self._col.delete(ids=existing["ids"])
        return len(existing["ids"])

    def search(self, query: str, n: int = 5, where: dict | None = None) -> list[dict]:
        if self._col.count() == 0:
            return []
        fetch_n = min(n * 3, self._col.count())
        query_kwargs: dict = {"query_texts": [query], "n_results": fetch_n}
        if where is not None:
            query_kwargs["where"] = where
        results = self._col.query(**query_kwargs)
        out = []
        seen_docs: set[str] = set()
        for i in range(len(results["ids"][0])):
            doc_id = results["metadatas"][0][i].get("doc_id", "")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            out.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
            if len(out) >= n:
                break
        return out

    def list_documents(self) -> list[dict]:
        all_data = self._col.get(include=["metadatas"])
        docs: dict[str, dict] = {}
        for m in all_data["metadatas"]:
            doc_id = m.get("doc_id", "?")
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": m.get("title", ""),
                    "source": m.get("source", ""),
                    "category": m.get("category", ""),
                    "chunks": 0,
                }
            docs[doc_id]["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def stats(self) -> dict:
        docs = self.list_documents()
        categories = {}
        for d in docs:
            cat = d.get("category", "uncategorized") or "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_chunks": self._col.count(),
            "total_documents": len(docs),
            "categories": categories,
            "model": self._model_name,
        }
