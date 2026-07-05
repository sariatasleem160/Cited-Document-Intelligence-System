"""Load documents, chunk, embed, and persist to the vector index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from chunking import Chunk, chunk_markdown, chunk_pdf_pages
from embeddings import embed_texts
from retrieve import VectorStore

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".pdf"}


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i, text))
    return pages


def load_document(path: Path, *, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return chunk_markdown(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if suffix == ".pdf":
        pages = _read_pdf_pages(path)
        return chunk_pdf_pages(
            pages,
            source=str(path),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    raise ValueError(f"Unsupported file type: {path.suffix}")


def chunks_to_metadata(chunks: list[Chunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "start_char": chunk.start_char,
            "end_char": chunk.end_char,
        }
        for chunk in chunks
    ]


def ingest_directory(
    docs_dir: Path,
    index_dir: Path,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    reset: bool = False,
) -> int:
    docs_dir = Path(docs_dir)
    index_dir = Path(index_dir)
    store = VectorStore(index_dir)

    existing_meta: list[dict[str, Any]] = []
    existing_vectors = None
    if not reset and (index_dir / "vectors.npy").exists():
        store.load()
        existing_meta = store._metadata
        existing_vectors = store._vectors

    all_chunks: list[Chunk] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            all_chunks.extend(load_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap))

    if not all_chunks:
        if existing_meta:
            return len(existing_meta)
        raise FileNotFoundError(f"No supported documents found in {docs_dir}")

    new_meta = chunks_to_metadata(all_chunks)
    new_vectors = embed_texts([row["text"] for row in new_meta])

    if existing_meta and not reset:
        import numpy as np

        combined_meta = existing_meta + new_meta
        combined_vectors = np.vstack([existing_vectors, new_vectors])
    else:
        combined_meta = new_meta
        combined_vectors = new_vectors

    store.save(combined_vectors, combined_meta)
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the vector index.")
    parser.add_argument("--docs", type=Path, default=Path("docs"), help="Directory containing PDF/Markdown files.")
    parser.add_argument("--index", type=Path, default=Path("data/index"), help="Vector index output directory.")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--reset", action="store_true", help="Replace the existing index.")
    args = parser.parse_args()

    count = ingest_directory(
        args.docs,
        args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        reset=args.reset,
    )
    print(f"Ingested {count} chunks into {args.index}")


if __name__ == "__main__":
    main()
