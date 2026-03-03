from __future__ import annotations

import csv
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from kselect.exceptions import LoaderError

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
_MIN_FILE_SIZE_DEFAULT = 64          # bytes
_MAX_FILE_SIZE_DEFAULT = 50.0        # MB


class BaseLoader(ABC):
    @abstractmethod
    def load(self) -> list[tuple[str, dict]]:
        """Returns list of (raw_text, metadata_dict) pairs. One pair per document."""


# ── FolderLoader ──────────────────────────────────────────────────────────────


class FolderLoader(BaseLoader):
    """
    Recursively walks a directory tree. Dispatches by file extension:
      .pdf   → pypdf.PdfReader (page text concatenated)
      .docx  → python-docx Document (paragraph text concatenated)
      .txt   → plain utf-8 read
      .md    → plain read (preserve markdown syntax — do not strip)

    extract_tables=True: routes .pdf through unstructured.partition_pdf()
    for table extraction before text extraction.

    Skips files smaller than min_file_size_bytes (default 64 B),
    larger than max_file_size_mb (default 50 MB), or with unsupported extensions.
    Logs a DEBUG warning for each skipped file.

    max_docs: stop after loading this many documents (for testing).
    """

    def __init__(
        self,
        path: str,
        extract_tables: bool = False,
        min_file_size_bytes: int = _MIN_FILE_SIZE_DEFAULT,
        max_file_size_mb: float = _MAX_FILE_SIZE_DEFAULT,
        max_docs: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._extract_tables = extract_tables
        self._min_bytes = min_file_size_bytes
        self._max_bytes = int(max_file_size_mb * 1024 * 1024)
        self._max_docs = max_docs

    def load(self) -> list[tuple[str, dict]]:
        results: list[tuple[str, dict]] = []
        for file_path in sorted(self._path.rglob("*")):
            if not file_path.is_file():
                continue
            if self._max_docs is not None and len(results) >= self._max_docs:
                break

            size = file_path.stat().st_size
            ext = file_path.suffix.lower()

            if ext not in _SUPPORTED_EXTENSIONS:
                logger.debug("FolderLoader: skipping unsupported extension %s", file_path)
                continue
            if size < self._min_bytes:
                logger.debug("FolderLoader: skipping too-small file %s (%d B)", file_path, size)
                continue
            if size > self._max_bytes:
                logger.debug("FolderLoader: skipping too-large file %s (%.1f MB)", file_path, size / 1e6)
                continue

            try:
                text = self._extract(file_path, ext)
            except Exception as exc:
                raise LoaderError(f"Failed to read {file_path}: {exc}") from exc

            meta: dict[str, Any] = {"source_file": str(file_path)}
            results.append((text, meta))

        return results

    def _extract(self, file_path: Path, ext: str) -> str:
        if ext == ".pdf":
            return self._read_pdf(file_path)
        if ext == ".docx":
            return self._read_docx(file_path)
        # .txt and .md — plain UTF-8
        return file_path.read_text(encoding="utf-8", errors="replace")

    def _read_pdf(self, file_path: Path) -> str:
        if self._extract_tables:
            try:
                from unstructured.partition.pdf import partition_pdf  # type: ignore[import-untyped]
                elements = partition_pdf(filename=str(file_path))
                return "\n\n".join(str(el) for el in elements)
            except ImportError:
                logger.warning(
                    "unstructured not installed; falling back to pypdf for %s. "
                    "Install with: pip install 'kselect[unstructured]'",
                    file_path,
                )

        try:
            from pypdf import PdfReader  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LoaderError(
                "pypdf is required to load PDF files. Install it with: pip install pypdf"
            ) from exc

        reader = PdfReader(str(file_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())

    def _read_docx(self, file_path: Path) -> str:
        try:
            from docx import Document  # type: ignore[import-untyped]
        except ImportError as exc:
            raise LoaderError(
                "python-docx is required to load .docx files. "
                "Install it with: pip install python-docx"
            ) from exc

        doc = Document(str(file_path))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── CSVLoader ─────────────────────────────────────────────────────────────────


class CSVLoader(BaseLoader):
    """
    text_col is required — raises LoaderError if column not found.
    All other columns not in the metadata list are ignored.
    metadata: list of column names to include as metadata.
              If None, includes all columns except text_col.
    vector_col: if provided, parse this column as list[float] embedding.
                IngestionPipeline skips re-embedding when vector_col is present.
    """

    def __init__(
        self,
        path: str,
        text_col: str,
        metadata: list[str] | None = None,
        vector_col: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._text_col = text_col
        self._metadata = metadata
        self._vector_col = vector_col

    def load(self) -> list[tuple[str, dict]]:
        results: list[tuple[str, dict]] = []
        try:
            with self._path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is None:
                    raise LoaderError(f"CSV file is empty or has no header: {self._path}")
                fieldnames = list(reader.fieldnames)

                if self._text_col not in fieldnames:
                    raise LoaderError(
                        f"text_col {self._text_col!r} not found in CSV columns: {fieldnames}"
                    )

                # Determine metadata columns
                if self._metadata is not None:
                    meta_cols = [c for c in self._metadata if c in fieldnames]
                else:
                    meta_cols = [c for c in fieldnames if c != self._text_col]

                for row in reader:
                    text = row[self._text_col]
                    meta: dict[str, Any] = {
                        "source_file": str(self._path),
                        **{col: row[col] for col in meta_cols},
                    }
                    if self._vector_col and self._vector_col in row:
                        raw = row[self._vector_col]
                        try:
                            meta["__precomputed_embedding"] = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            pass
                    results.append((text, meta))

        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"Failed to read CSV {self._path}: {exc}") from exc

        return results


# ── JSONLoader ────────────────────────────────────────────────────────────────


class JSONLoader(BaseLoader):
    """
    Handles both:
    - JSON array:  top-level list of objects
    - JSONL:       one JSON object per line (auto-detected by attempting
                   json.loads on the first line)
    text_key: required. The key whose value is the document text.
    metadata: list of keys to include as metadata.
              If None, includes all keys except text_key.
    """

    def __init__(
        self,
        path: str,
        text_key: str,
        metadata: list[str] | None = None,
    ) -> None:
        self._path = Path(path)
        self._text_key = text_key
        self._metadata = metadata

    def load(self) -> list[tuple[str, dict]]:
        try:
            content = self._path.read_text(encoding="utf-8")
        except Exception as exc:
            raise LoaderError(f"Cannot read {self._path}: {exc}") from exc

        # Auto-detect JSONL vs JSON array
        records = self._parse(content)

        results: list[tuple[str, dict]] = []
        for i, record in enumerate(records):
            if self._text_key not in record:
                raise LoaderError(
                    f"text_key {self._text_key!r} not found in record {i} "
                    f"of {self._path}. Available keys: {list(record.keys())}"
                )
            text = str(record[self._text_key])
            if self._metadata is not None:
                meta_keys = self._metadata
            else:
                meta_keys = [k for k in record if k != self._text_key]

            meta: dict[str, Any] = {
                "source_file": str(self._path),
                **{k: record[k] for k in meta_keys if k in record},
            }
            results.append((text, meta))

        return results

    def _parse(self, content: str) -> list[dict]:
        # Try JSONL first: if first non-empty line is a valid JSON object
        lines = [ln for ln in content.splitlines() if ln.strip()]
        if not lines:
            return []

        try:
            first = json.loads(lines[0])
            if isinstance(first, dict):
                # Looks like JSONL
                records = []
                for ln in lines:
                    try:
                        obj = json.loads(ln)
                        if isinstance(obj, dict):
                            records.append(obj)
                    except json.JSONDecodeError:
                        pass
                return records
        except json.JSONDecodeError:
            pass

        # Fall back to full JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [r for r in data if isinstance(r, dict)]
            if isinstance(data, dict):
                return [data]
        except json.JSONDecodeError as exc:
            raise LoaderError(f"Cannot parse JSON from {self._path}: {exc}") from exc

        return []
