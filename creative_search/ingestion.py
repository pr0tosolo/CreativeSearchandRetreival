"""File ingestion and conversion helpers.

The helper functions in this module deliberately keep their
interfaces small so they can be orchestrated inside the pipeline
module or reused in notebooks.  They follow the breakdown
outlined in the specification shared via the conversation link.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Sequence, Set

try:  # pragma: no cover - optional dependency
    from markitdown import MarkItDown

    _MARKITDOWN_AVAILABLE = True
except ImportError:  # pragma: no cover - markitdown is optional
    MarkItDown = None  # type: ignore
    _MARKITDOWN_AVAILABLE = False


def list_files(
    base_dirs: Sequence[str | Path],
    extensions: Optional[Set[str]] = None,
) -> Iterator[Path]:
    """Yield files under ``base_dirs`` optionally filtered by ``extensions``.

    Parameters
    ----------
    base_dirs:
        A sequence of root directories to traverse.
    extensions:
        Lower-case file extensions to include (e.g. {"pdf", "pptx"}).
        When ``None`` all files are yielded.
    """

    normalized_exts = {ext.lower().lstrip(".") for ext in extensions} if extensions else None

    for base in base_dirs:
        base_path = Path(base).expanduser().resolve()
        if not base_path.exists():
            continue
        for path in base_path.rglob("*"):
            if not path.is_file():
                continue
            if normalized_exts:
                suffix = path.suffix.lower().lstrip(".")
                if suffix not in normalized_exts:
                    continue
            yield path


def convert_file_to_markdown(path: str | Path, *, encoding: str = "utf-8") -> str:
    """Convert ``path`` to Markdown using MarkItDown when available.

    The real system is expected to support many binary formats via
    `markitdown`.  For local development the dependency is optional;
    if it is missing we fall back to returning the raw text for plain
    text / Markdown documents, which keeps the pipeline usable for
    smoke tests.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    if _MARKITDOWN_AVAILABLE and MarkItDown is not None:
        converter = MarkItDown(enable_plugins=False)
        result = converter.convert(str(source))
        return result.text_content

    # Fallback for plain-text friendly formats
    try:
        return source.read_text(encoding=encoding)
    except UnicodeDecodeError as exc:  # pragma: no cover - depends on file contents
        raise RuntimeError(
            "markitdown is not installed and the file could not be decoded as text"
        ) from exc


def ingest_and_convert(file_paths: Iterable[str | Path]) -> Dict[str, str]:
    """Convert the given files and return a mapping ``path -> markdown``.

    Conversion errors are logged but do not stop the pipeline, mirroring
    the fault-tolerant guidance from the original plan.
    """

    markdown_map: Dict[str, str] = {}
    for path in file_paths:
        try:
            markdown_map[str(path)] = convert_file_to_markdown(path)
        except Exception as error:  # pragma: no cover - logging path
            print(f"[ingestion] Failed to convert {path}: {error}")
    return markdown_map


def download_from_dingtalk(
    token: str,
    channel_id: str,
    dest_dir: str | Path,
) -> Mapping[str, str]:
    """Placeholder for pulling attachments from DingTalk.

    The plan calls for integrating DingTalk but the concrete API
    contract depends on an organization's credentials.  This
    placeholder returns an empty mapping and documents where real
    integration logic should live.
    """

    _ = (token, channel_id)  # hint that the parameters are intentionally unused
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    return {}
