"""Embedding utilities using sentence-transformers."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(texts: Sequence[str], *, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model(model_name)
    vectors = model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return vectors.astype(np.float32)


def embed_query(query: str, *, model_name: str = DEFAULT_MODEL) -> np.ndarray:
    return embed_texts([query], model_name=model_name)[0]
