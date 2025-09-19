# CreativeSearchandRetreival

This repository contains a reference implementation of the plan outlined in the
shared conversation for building a *Creative Search and Retrieval* pipeline.
The code base is structured as a small Python package (`creative_search`) that
orchestrates the following stages:

1. **Ingestion & conversion** – traverse configured server folders, optionally
   pull new files from DingTalk, and convert supported assets to Markdown using
   [MarkItDown](https://github.com/microsoft/markitdown) when the library is
   installed. Plain-text files fall back to direct reading.
2. **Metadata extraction** – apply lightweight heuristics (and spaCy when
   available) to identify concepts, colours, materials, product names, prices,
   and other named entities inside the converted Markdown content.
3. **Embedding generation** – use `sentence-transformers` to create dense vector
   representations, or a deterministic hash-based embedding when the library is
   not available. The abstraction keeps local development lightweight while
   enabling higher quality vectors in production.
4. **Vector storage** – push embeddings into a FAISS index when the dependency
   exists, or into a pure NumPy fallback. Metadata is persisted in SQLite via
   SQLAlchemy, with an in-memory fallback for minimal environments.
5. **Retrieval API** – expose a FastAPI service that performs semantic search
   over the indexed documents and returns rich metadata for each match.

## Repository layout

```
creative_search/
├── __init__.py          # Package export convenience
├── ingestion.py         # File listing, DingTalk placeholder, Markdown conversion
├── extraction.py        # Heuristics for structured metadata extraction
├── embedding.py         # SentenceTransformer wrapper with hash fallback
├── models.py            # SQLAlchemy model definitions and fallbacks
├── storage.py           # Vector index management and search utilities
├── pipeline.py          # CLI and orchestration helpers for the ingestion run
└── main.py              # FastAPI application exposing the search endpoint
```

## Installation

Create a virtual environment and install the core dependencies:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install sqlalchemy numpy
```

Optional enhancements:

- `pip install markitdown` – enable rich document conversion.
- `pip install sentence-transformers` – use high-quality text embeddings.
- `pip install faiss-cpu` – leverage the native FAISS index.
- `pip install fastapi uvicorn pydantic` – run the HTTP API.
- `pip install spacy && python -m spacy download en_core_web_sm` – improve
  entity recognition during extraction.

The code automatically falls back to lighter-weight implementations when these
packages are absent, which makes local experimentation easy while still keeping
an upgrade path for production deployments.

## Running the pipeline

The pipeline can be executed via the command line. The example below scans two
folders for PDF and PowerPoint files and populates the vector index:

```sh
python -m creative_search.pipeline \
  --dirs /srv/design-assets /srv/shared/briefs \
  --extensions pdf pptx docx
```

Additional options:

- `--ding-token` / `--ding-channel` – (placeholder) configuration for retrieving
  attachments from DingTalk and dropping them into the first directory.
- `--database-url` – custom SQLAlchemy URL. Defaults to `sqlite:///metadata.db`.

## Starting the API

After running the ingestion pipeline you can start the FastAPI service:

```sh
uvicorn creative_search.main:app --host 0.0.0.0 --port 8000
```

Query the endpoint with `GET /search?query=red+leather+jacket` to receive the
closest matches (including extracted metadata). A simple health check is
exposed at `GET /healthz`.

## Development notes

- The storage layer exposes an in-memory fallback when SQLAlchemy is not
  installed, which is useful for unit tests or rapid prototyping.
- The hash-based embedding fallback ensures deterministic behaviour without
  requiring heavyweight ML downloads. Replace it with `sentence-transformers`
  in production for higher-quality results.
- DingTalk integration is represented by a stub in `ingestion.py`. Replace the
  placeholder with real API calls once credentials are available.

## License

MIT
