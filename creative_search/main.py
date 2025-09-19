"""FastAPI application exposing semantic search endpoints."""
from __future__ import annotations

from typing import List, Optional

try:  # pragma: no cover - optional dependency
    from fastapi import FastAPI, Query
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - make the error actionable
    raise ImportError(
        "FastAPI and Pydantic are required for the API. Install them via 'pip install fastapi pydantic'."
    ) from exc

from . import storage


class SearchResult(BaseModel):
    score: float
    file_path: str
    file_url: Optional[str] = None
    concepts: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    prices: List[str] = Field(default_factory=list)
    product_types: List[str] = Field(default_factory=list)
    clients: List[str] = Field(default_factory=list)
    product_lines: List[str] = Field(default_factory=list)
    proper_nouns: List[str] = Field(default_factory=list)


app = FastAPI(title="Creative Search and Retrieval")


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/search", response_model=List[SearchResult])
def search(query: str = Query(..., description="Search phrase"), k: int = Query(5, ge=1, le=50)):
    results = storage.search_similar(query, k)
    payload: List[SearchResult] = []
    for result in results:
        metadata = result["metadata"]
        payload.append(
            SearchResult(
                score=result["score"],
                file_path=metadata.get("file_path"),
                file_url=metadata.get("file_url"),
                concepts=metadata.get("concepts", []),
                colors=metadata.get("colors", []),
                materials=metadata.get("materials", []),
                prices=metadata.get("prices", []),
                product_types=metadata.get("product_types", []),
                clients=metadata.get("clients", []),
                product_lines=metadata.get("product_lines", []),
                proper_nouns=metadata.get("proper_nouns", []),
            )
        )
    return payload
