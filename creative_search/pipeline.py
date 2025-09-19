"""Orchestrates ingestion, conversion, and indexing."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence, Set

from . import ingestion, storage


def run_pipeline(
    server_dirs: Sequence[str | Path],
    *,
    extensions: Optional[Set[str]] = None,
    ding_token: Optional[str] = None,
    ding_channel: Optional[str] = None,
    database_url: Optional[str] = None,
) -> None:
    """Run the end-to-end ingestion workflow."""

    if not server_dirs:
        raise ValueError("At least one server directory must be provided")

    downloaded: Dict[str, str] = {}
    primary_dir = Path(server_dirs[0]).expanduser()
    if ding_token and ding_channel:
        downloaded = ingestion.download_from_dingtalk(
            ding_token,
            ding_channel,
            dest_dir=primary_dir,
        )

    file_paths = list(ingestion.list_files(server_dirs, extensions=extensions))
    markdown_map = ingestion.ingest_and_convert(file_paths)

    session_factory = None
    if database_url is not None:
        session_factory = storage.get_session_factory(database_url)

    storage.add_documents_to_index(
        markdown_map,
        file_url_map=downloaded,
        session_factory=session_factory,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Creative Search ingestion pipeline")
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="One or more directories that contain creative assets",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        help="Optional set of file extensions to include (e.g. pdf pptx docx)",
    )
    parser.add_argument("--ding-token", help="DingTalk API token", dest="ding_token")
    parser.add_argument("--ding-channel", help="DingTalk channel identifier", dest="ding_channel")
    parser.add_argument("--database-url", help="SQLAlchemy database URL", dest="database_url")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_pipeline(
        args.dirs,
        extensions=set(args.extensions) if args.extensions else None,
        ding_token=args.ding_token,
        ding_channel=args.ding_channel,
        database_url=args.database_url,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
