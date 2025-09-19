"""Utilities for building a creative document search pipeline.

This package bundles helper modules for ingesting rich-media
files, extracting structured metadata, generating embeddings,
and exposing a simple retrieval API. The implementation follows
the plan outlined in the shared conversation for the
"Creative Search and Retrieval" prototype.
"""

from . import ingestion, extraction, embedding, models, storage, pipeline

__all__ = [
    "ingestion",
    "extraction",
    "embedding",
    "models",
    "storage",
    "pipeline",
]
