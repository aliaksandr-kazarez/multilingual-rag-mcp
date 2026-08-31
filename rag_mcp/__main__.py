import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "index":
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
