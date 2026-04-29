import sys

from pipeline import RAGPipeline


def main() -> None:
    rag = RAGPipeline()

    print("Ingesting documents...")
    rag.ingest()

    if rag.store.is_empty:
        sys.exit(0)

    print("\nReady. Type your question or 'quit' to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        result = rag.query(question, verbose=True)

        print(f"\nAssistant: {result['answer']}")

        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  • {s['file']}, page {s['page']}")

        print(f"  ({result.get('hops', 1)} hop(s))\n")


if __name__ == "__main__":
    main()