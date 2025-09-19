"""Vector index management and persistence helpers."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .embedding import generate_embedding, get_backend
from .extraction import extract_entities
from .models import (
    DocumentMetadata,
    SQLALCHEMY_AVAILABLE,
    create_session_factory,
)

try:  # pragma: no cover - optional dependency
    import faiss

    FAISS_AVAILABLE = True
except Exception:  # pragma: no cover - degrade gracefully
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


class _NumpyFlatIndex:
    """Minimal FAISS-like index for environments without the real library."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.zeros((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._vectors = np.vstack([self._vectors, arr])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        query_arr = np.asarray(query, dtype=np.float32)
        if query_arr.ndim == 1:
            query_arr = query_arr.reshape(1, -1)
        if self.ntotal == 0:
            distances = np.full((query_arr.shape[0], k), np.inf, dtype=np.float32)
            indices = -np.ones((query_arr.shape[0], k), dtype=np.int64)
            return distances, indices

        dists = np.linalg.norm(self._vectors[None, :, :] - query_arr[:, None, :], axis=2)
        k_eff = min(k, self.ntotal)
        order = np.argsort(dists, axis=1)[:, :k_eff]
        distances = np.take_along_axis(dists, order, axis=1)

        if k_eff == k:
            return distances.astype(np.float32), order.astype(np.int64)

        padded_dist = np.full((query_arr.shape[0], k), np.inf, dtype=np.float32)
        padded_idx = -np.ones((query_arr.shape[0], k), dtype=np.int64)
        padded_dist[:, :k_eff] = distances
        padded_idx[:, :k_eff] = order
        return padded_dist, padded_idx


class VectorIndex:
    """Abstraction that hides whether FAISS is available."""

    def __init__(self, dimension: int) -> None:
        if FAISS_AVAILABLE:  # pragma: no cover - requires faiss runtime
            self._index = faiss.IndexFlatL2(dimension)
        else:
            self._index = _NumpyFlatIndex(dimension)

    @property
    def ntotal(self) -> int:
        if FAISS_AVAILABLE:  # pragma: no branch - tiny helper
            return int(self._index.ntotal)
        return int(self._index.ntotal)

    def add(self, vectors: np.ndarray) -> None:
        if FAISS_AVAILABLE:
            self._index.add(np.asarray(vectors, dtype=np.float32))
        else:
            self._index.add(vectors)

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if FAISS_AVAILABLE:
            return self._index.search(np.asarray(query, dtype=np.float32), k)
        return self._index.search(query, k)


_INDEX: VectorIndex | None = None
_SESSION_FACTORY = None
_IN_MEMORY_METADATA: Dict[int, DocumentMetadata] = {}


def get_index() -> VectorIndex:
    global _INDEX
    if _INDEX is None:
        dimension = get_backend().dimension
        _INDEX = VectorIndex(dimension)
    return _INDEX


def get_session_factory(database_url: Optional[str] = None):
    global _SESSION_FACTORY
    if _SESSION_FACTORY is None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is required for persistence. Install it or run in memory only."
            )
        _SESSION_FACTORY = create_session_factory(database_url or "sqlite:///metadata.db")
    return _SESSION_FACTORY


def reset_state() -> None:
    """Reset globals – handy for unit tests."""

    global _INDEX, _SESSION_FACTORY, _IN_MEMORY_METADATA
    _INDEX = None
    _SESSION_FACTORY = None
    _IN_MEMORY_METADATA = {}


def add_documents_to_index(
    markdown_map: Dict[str, str],
    *,
    file_url_map: Optional[Dict[str, str]] = None,
    session_factory=None,
) -> None:
    """Extract metadata for ``markdown_map`` and add them to the vector index."""

    index = get_index()
    file_url_map = file_url_map or {}

    if SQLALCHEMY_AVAILABLE:
        session_factory = session_factory or get_session_factory()
        session = session_factory()
        try:
            for path, markdown in markdown_map.items():
                metadata = extract_entities(markdown)
                record = (
                    session.query(DocumentMetadata)
                    .filter_by(file_path=path)
                    .one_or_none()
                )

                if record is None:
                    embedding = generate_embedding(markdown)
                    vector = np.asarray([embedding], dtype=np.float32)
                    embedding_index = index.ntotal
                    index.add(vector)
                    record = DocumentMetadata(
                        file_path=path,
                        file_url=file_url_map.get(path),
                        concepts=metadata.concepts,
                        colors=metadata.colors,
                        materials=metadata.materials,
                        prices=metadata.prices,
                        product_types=metadata.product_types,
                        clients=metadata.clients,
                        product_lines=metadata.product_lines,
                        proper_nouns=metadata.proper_nouns,
                        embedding_index=embedding_index,
                    )
                    session.add(record)
                else:
                    record.file_url = file_url_map.get(path)
                    record.concepts = metadata.concepts
                    record.colors = metadata.colors
                    record.materials = metadata.materials
                    record.prices = metadata.prices
                    record.product_types = metadata.product_types
                    record.clients = metadata.clients
                    record.product_lines = metadata.product_lines
                    record.proper_nouns = metadata.proper_nouns
            session.commit()
        finally:
            session.close()
    else:  # fallback: store metadata in memory only
        for path, markdown in markdown_map.items():
            metadata = extract_entities(markdown)
            existing_index = next(
                (idx for idx, meta in _IN_MEMORY_METADATA.items() if meta.file_path == path),
                None,
            )
            if existing_index is None:
                embedding = generate_embedding(markdown)
                vector = np.asarray([embedding], dtype=np.float32)
                embedding_index = index.ntotal
                index.add(vector)
                _IN_MEMORY_METADATA[embedding_index] = DocumentMetadata(
                    file_path=path,
                    file_url=file_url_map.get(path),
                    concepts=metadata.concepts,
                    colors=metadata.colors,
                    materials=metadata.materials,
                    prices=metadata.prices,
                    product_types=metadata.product_types,
                    clients=metadata.clients,
                    product_lines=metadata.product_lines,
                    proper_nouns=metadata.proper_nouns,
                    embedding_index=embedding_index,
                )
            else:
                record = _IN_MEMORY_METADATA[existing_index]
                record.file_url = file_url_map.get(path)
                record.concepts = metadata.concepts
                record.colors = metadata.colors
                record.materials = metadata.materials
                record.prices = metadata.prices
                record.product_types = metadata.product_types
                record.clients = metadata.clients
                record.product_lines = metadata.product_lines
                record.proper_nouns = metadata.proper_nouns


def search_similar(query: str, k: int = 5) -> List[dict]:
    """Return metadata records ordered by vector similarity."""

    index = get_index()
    if index.ntotal == 0:
        return []

    query_emb = generate_embedding(query)
    distances, indices = index.search(np.asarray([query_emb]), k)

    results: List[dict] = []
    if SQLALCHEMY_AVAILABLE:
        session = get_session_factory()()
        try:
            for distance, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                record = (
                    session.query(DocumentMetadata)
                    .filter_by(embedding_index=int(idx))
                    .one_or_none()
                )
                if record is None:
                    continue
                results.append({
                    "score": float(distance),
                    "metadata": record.as_dict(),
                })
        finally:
            session.close()
    else:
        for distance, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            record = _IN_MEMORY_METADATA.get(int(idx))
            if record is None:
                continue
            results.append({
                "score": float(distance),
                "metadata": record.as_dict(),
            })
    return results
