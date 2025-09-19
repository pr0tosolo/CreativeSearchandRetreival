"""Vector embedding utilities."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class EmbeddingBackend:
    """Wrapper that hides the optional SentenceTransformer dependency."""

    model_name: str = DEFAULT_MODEL_NAME

    def __post_init__(self) -> None:
        try:  # pragma: no cover - optional heavy dependency
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
            self._backend = "sentence-transformers"
        except Exception:  # pragma: no cover - handle missing library
            self._model = None
            self._dimension = 384
            self._backend = "hashed"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def backend(self) -> str:
        return self._backend

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if self._model is not None:
            return np.asarray(
                self._model.encode(list(texts), show_progress_bar=False), dtype=np.float32
            )
        return np.vstack([_hash_to_unit_vector(text, self._dimension) for text in texts])


_BACKEND: EmbeddingBackend | None = None


def get_backend(model_name: str = DEFAULT_MODEL_NAME) -> EmbeddingBackend:
    global _BACKEND
    if _BACKEND is None or _BACKEND.model_name != model_name:
        _BACKEND = EmbeddingBackend(model_name)
    return _BACKEND


def generate_embedding(text: str) -> np.ndarray:
    backend = get_backend()
    return backend.encode([text])[0]


def batch_generate_embedding(texts: Iterable[str]) -> np.ndarray:
    backend = get_backend()
    return backend.encode(list(texts))


def _hash_to_unit_vector(text: str, dimension: int) -> np.ndarray:
    """Create a deterministic pseudo-embedding when real models are unavailable."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat digest to cover desired dimensionality
    needed = dimension * 4
    data = (digest * (needed // len(digest) + 1))[: needed]
    vector = np.frombuffer(data, dtype=np.uint32).astype(np.float32)
    vector = vector[:dimension]
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return np.zeros(dimension, dtype=np.float32)
    return vector / norm
