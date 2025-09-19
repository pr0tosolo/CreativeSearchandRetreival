"""Database models used by the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:  # pragma: no cover - optional dependency
    from sqlalchemy import JSON, Column, Integer, String, create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker

    SQLALCHEMY_AVAILABLE = True
except Exception:  # pragma: no cover - degrade gracefully
    JSON = Column = Integer = String = create_engine = None  # type: ignore
    sessionmaker = None  # type: ignore
    declarative_base = lambda: None  # type: ignore
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()

    class DocumentMetadata(Base):
        __tablename__ = "documents"

        id = Column(Integer, primary_key=True)
        file_path = Column(String, unique=True, nullable=False)
        file_url = Column(String, nullable=True)
        concepts = Column(JSON, nullable=True)
        colors = Column(JSON, nullable=True)
        materials = Column(JSON, nullable=True)
        prices = Column(JSON, nullable=True)
        product_types = Column(JSON, nullable=True)
        clients = Column(JSON, nullable=True)
        product_lines = Column(JSON, nullable=True)
        proper_nouns = Column(JSON, nullable=True)
        embedding_index = Column(Integer, nullable=False)

        def as_dict(self) -> dict:
            return {
                "file_path": self.file_path,
                "file_url": self.file_url,
                "concepts": self.concepts or [],
                "colors": self.colors or [],
                "materials": self.materials or [],
                "prices": self.prices or [],
                "product_types": self.product_types or [],
                "clients": self.clients or [],
                "product_lines": self.product_lines or [],
                "proper_nouns": self.proper_nouns or [],
            }

    def create_session_factory(database_url: str = "sqlite:///metadata.db"):
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)
        return sessionmaker(bind=engine)

else:
    Base = None

    @dataclass
    class DocumentMetadata:  # pragma: no cover - fallback container
        file_path: str
        file_url: str | None = None
        concepts: Sequence[str] = field(default_factory=list)
        colors: Sequence[str] = field(default_factory=list)
        materials: Sequence[str] = field(default_factory=list)
        prices: Sequence[str] = field(default_factory=list)
        product_types: Sequence[str] = field(default_factory=list)
        clients: Sequence[str] = field(default_factory=list)
        product_lines: Sequence[str] = field(default_factory=list)
        proper_nouns: Sequence[str] = field(default_factory=list)
        embedding_index: int = -1

        def as_dict(self) -> dict:
            return {
                "file_path": self.file_path,
                "file_url": self.file_url,
                "concepts": list(self.concepts),
                "colors": list(self.colors),
                "materials": list(self.materials),
                "prices": list(self.prices),
                "product_types": list(self.product_types),
                "clients": list(self.clients),
                "product_lines": list(self.product_lines),
                "proper_nouns": list(self.proper_nouns),
            }

    def create_session_factory(database_url: str = "sqlite:///metadata.db"):
        raise ImportError(
            "SQLAlchemy is required for persistent storage. Install it via 'pip install SQLAlchemy'."
        )
