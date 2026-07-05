"""Simple CLI for the cited document assistant."""

from __future__ import annotations

import argparse
from pathlib import Path

from answer import answer_question
from ingest import ingest_directory
from retrieve import VectorStore


def ask(question: str, index_dir: Path, *, top_k: int = 5) -> str:
    store = VectorStore(index_dir)
    chunks = store.retrieve(question, top_k=top_k)
    result = answer_question(question, chunks)
    return result.answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Cited Document Intelligence System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector index.")
    ingest_parser.add_argument("--docs", type=Path, default=Path("docs"))
    ingest_parser.add_argument("--index", type=Path, default=Path("data/index"))
    ingest_parser.add_argument("--chunk-size", type=int, default=500)
    ingest_parser.add_argument("--chunk-overlap", type=int, default=100)
    ingest_parser.add_argument("--reset", action="store_true")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested documents.")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--index", type=Path, default=Path("data/index"))
    ask_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    if args.command == "ingest":
        count = ingest_directory(
            args.docs,
            args.index,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            reset=args.reset,
        )
        print(f"Ingested {count} chunks.")
        return

    if args.command == "ask":
        print(ask(args.question, args.index, top_k=args.top_k))


if __name__ == "__main__":
    main()
