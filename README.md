# rag-mcp

Multilingual RAG MCP server for local document search. Query in one language, find content in another.

Built for the common case where you talk to AI agents in English but your documents are in Russian (or any other language). The multilingual embedding model maps semantically similar concepts across 50+ languages to the same vector space — no translation step needed.

## Install

### pip from PyPI

```bash
pip install multilingual-rag-mcp
```

### One-liner with uvx (no install needed)

```bash
uvx --from multilingual-rag-mcp rag-mcp index ./docs/
```

### pip from GitHub (latest)

```bash
pip install git+https://github.com/aliaksandr-kazarez/rag-mcp.git
```

## Quick start

### 1. Index your documents

```bash
rag-mcp index ~/documents/
```

### 2. Add to Claude Code

```bash
claude mcp add rag -- uvx --from multilingual-rag-mcp rag-mcp
```

Set the document paths via env vars:

```bash
claude mcp add rag \
  -e RAG_DOCS=$HOME/documents \
  -- uvx --from multilingual-rag-mcp rag-mcp
```

If installed locally (pip install), use the simpler form:

```bash
claude mcp add rag -e RAG_DOCS=$HOME/documents -- rag-mcp
```

### 3. Search

From Claude Code, the `search` tool handles cross-lingual queries automatically:

- "search for protein recommendations" finds Russian articles about белок
- "найди рецепты" finds recipe content regardless of language

## MCP tools

| Tool | Description |
|------|-------------|
| `search(query, n=5)` | Semantic search across all indexed documents |
| `get_document(path)` | Retrieve full document content |
| `list_documents(filter?)` | List indexed documents, optionally filtered |
| `reindex()` | Re-index all configured document directories |
| `stats()` | Index statistics (documents, chunks, categories) |

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `RAG_DOCS` | — | Comma-separated paths to document directories |
| `RAG_DATA` | `~/.local/share/rag-mcp/` | Index storage location |
| `RAG_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (any fastembed-compatible model) |
| `RAG_CHUNK_SIZE` | `1000` | Maximum chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `200` | Overlap between chunks |

## Supported formats

Markdown (`.md`), plain text (`.txt`), PDF (`.pdf`), JSON (`.json`), CSV (`.csv`).

Markdown files with YAML frontmatter have their metadata (title, date, etc.) extracted automatically.

## CLI

```
rag-mcp index <dir> [<dir> ...]   Index documents
rag-mcp stats                     Print index statistics
rag-mcp                           Start MCP server (stdio)
rag-mcp --help                    Show help
rag-mcp --version                 Show version
```

## How it works

1. Documents are parsed and split into chunks (markdown-aware: respects headers, code blocks)
2. Each chunk is embedded with a multilingual model via [fastembed](https://github.com/qdrant/fastembed) (ONNX, no PyTorch needed)
3. Embeddings are stored in [ChromaDB](https://www.trychroma.com/) (local, file-based)
4. Queries are embedded with the same model and matched by cosine similarity
5. Cross-lingual retrieval works because the model places semantically similar text from different languages near each other in vector space

## License

MIT
