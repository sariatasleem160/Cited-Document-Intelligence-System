"""Vector retrieval over stored document chunks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from embeddings import embed_query


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    score: float


class VectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.meta_path = self.index_dir / "metadata.jsonl"
        self._vectors: np.ndarray | None = None
        self._metadata: list[dict[str, Any]] = []

    def load(self) -> None:
        if self.vectors_path.exists() and self.meta_path.exists():
            self._vectors = np.load(self.vectors_path)
            self._metadata = [
                json.loads(line) for line in self.meta_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ]
        else:
            self._vectors = np.zeros((0, 384), dtype=np.float32)
            self._metadata = []

    def save(self, vectors: np.ndarray, metadata: list[dict[str, Any]]) -> None:
        np.save(self.vectors_path, vectors.astype(np.float32))
        with self.meta_path.open("w", encoding="utf-8") as handle:
            for row in metadata:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._vectors = vectors.astype(np.float32)
        self._metadata = metadata

    @property
    def is_empty(self) -> bool:
        self.load()
        return self._vectors is None or len(self._metadata) == 0

    def retrieve(self, query: str, *, top_k: int = 5) -> list[RetrievedChunk]:
        self.load()
        if self._vectors is None or len(self._metadata) == 0:
            return []

        query_vec = embed_query(query)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        scores = (self._vectors @ query_vec) / norms
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[RetrievedChunk] = []
        for idx in top_indices:
            meta = self._metadata[int(idx)]
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    text=meta["text"],
                    source=meta["source"],
                    page=meta.get("page"),
                    score=float(scores[int(idx)]),
                )
            )
        return results
