"""Text chunking with overlap and metadata for citations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None = None
    start_char: int = 0
    end_char: int = 0


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(
    text: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    source: str = "unknown",
    page: int | None = None,
) -> list[Chunk]:
    """Split text into overlapping character-based chunks."""
    text = _normalize_whitespace(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    step = max(chunk_size - chunk_overlap, 1)
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{Path(source).stem}::p{page or 0}::c{index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    page=page,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1

        if end >= len(text):
            break
        start += step

    return chunks


def chunk_markdown(path: Path, *, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    return split_text(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        source=str(path),
        page=None,
    )


def chunk_pdf_pages(
    pages: Iterable[tuple[int, str]],
    source: str,
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page_num, page_text in pages:
        all_chunks.extend(
            split_text(
                page_text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source=source,
                page=page_num,
            )
        )
    return all_chunks
