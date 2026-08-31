import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help", "help"):
        print(
            "Usage: rag-mcp [command]\n"
            "\n"
            "Commands:\n"
            "  index <dir> [<dir> ...]   Index documents from directories\n"
            "  stats                     Print index statistics\n"
            "  (none)                    Start MCP server (stdio)\n"
            "\n"
            "Environment:\n"
            "  RAG_DOCS     Comma-separated document directories\n"
            "  RAG_DATA     Index storage (default: ~/.local/share/rag-mcp/)\n"
            "  RAG_MODEL    Embedding model (default: paraphrase-multilingual-MiniLM-L12-v2)\n"
        )
    elif len(sys.argv) > 1 and sys.argv[1] in ("-V", "--version"):
        from rag_mcp import __version__
        print(f"rag-mcp {__version__}")
    elif len(sys.argv) > 1 and sys.argv[1] == "index":
        from rag_mcp.ingest import index_cli
        index_cli(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "stats":
        from rag_mcp.store import VectorStore
        import json
        store = VectorStore()
        print(json.dumps(store.stats(), indent=2, ensure_ascii=False))
    else:
        from rag_mcp.server import mcp
        mcp.run()


if __name__ == "__main__":
    main()
