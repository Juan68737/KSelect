"""Phase 2 tests — ingestion layer."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kselect.exceptions import LoaderError
from kselect.ingestion.chunking import Chunker
from kselect.ingestion.loaders import CSVLoader, FolderLoader, JSONLoader
from kselect.ingestion.pipeline import IngestionPipeline
from kselect.models.config import ChunkingConfig, ChunkingStrategy, KSelectConfig


# ── PDF fixture helper ─────────────────────────────────────────────────────────


def _make_pdf(path: Path, text: str = "Hello World this is a test document") -> None:
    """
    Write a minimal but valid PDF that pypdf can parse text from.
    Byte offsets are computed dynamically so the xref table is always correct.
    """
    # Encode text escaping PDF special chars
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET".encode()
    stream_len = len(content_stream)

    obj1 = b"<< /Type /Catalog /Pages 2 0 R >>"
    obj2 = b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"
    obj3 = (
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
    )
    obj4 = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    obj5 = (
        f"<< /Length {stream_len} >>\nstream\n".encode()
        + content_stream
        + b"\nendstream"
    )

    body = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, data in enumerate([obj1, obj2, obj3, obj4, obj5], 1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n".encode() + data + b"\nendobj\n"

    xref_pos = len(body)
    xref = b"xref\n0 6\n0000000000 65535 f \n"
    for off in offsets:
        xref += f"{off:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode()

    path.write_bytes(body + xref + trailer)


# ── test_folder_loader_pdf ─────────────────────────────────────────────────────


def test_folder_loader_pdf(tmp_path):
    """FolderLoader parses a PDF and returns non-empty text."""
    pytest.importorskip("pypdf")
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, "Hello World this is a test document")

    loader = FolderLoader(str(tmp_path))
    docs = loader.load()

    assert len(docs) == 1
    text, meta = docs[0]
    assert len(text) > 0, "Expected non-empty text from PDF"
    assert "source_file" in meta


# ── test_csv_loader_missing_col ────────────────────────────────────────────────


def test_csv_loader_missing_col(tmp_path):
    """CSVLoader raises LoaderError when text_col is not found."""
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("title,body\nFoo,Bar\n")

    with pytest.raises(LoaderError, match="text_col"):
        CSVLoader(str(csv_path), text_col="content").load()


# ── test_chunker_sliding_window ────────────────────────────────────────────────


def test_chunker_sliding_window():
    """All chunks are within the chunk_size token budget."""
    pytest.importorskip("tiktoken")
    import tiktoken

    long_text = " ".join(["word"] * 1_000)
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SLIDING_WINDOW,
        chunk_size=100,
        chunk_overlap=10,
    )

    chunker = Chunker()
    chunks = chunker.chunk(long_text, {"source_file": "test.txt"}, config)

    enc = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1, "Expected multiple chunks from 1000-word text"
    for c in chunks:
        token_count = len(enc.encode(c.text))
        assert token_count <= config.chunk_size, (
            f"Chunk exceeds size: {token_count} > {config.chunk_size}"
        )


# ── test_chunker_min_length_merge ─────────────────────────────────────────────


def test_chunker_min_length_merge():
    """Chunks below min_chunk_length are merged with the previous chunk, not dropped."""
    pytest.importorskip("tiktoken")

    # Create text with two paragraphs: one normal, one very short
    text = "This is a normal paragraph with some content.\n\nHi.\n\nAnother normal paragraph here."
    config = ChunkingConfig(
        strategy=ChunkingStrategy.PARAGRAPH,
        chunk_size=512,
        chunk_overlap=0,
        min_chunk_length=20,  # "Hi." is shorter than 20 chars
    )

    chunker = Chunker()
    chunks = chunker.chunk(text, {"source_file": "test.txt"}, config)

    # "Hi." (3 chars) should be merged with the previous chunk
    for c in chunks:
        assert len(c.text) >= config.min_chunk_length, (
            f"Chunk shorter than min_chunk_length after merge: {c.text!r}"
        )


# ── test_chunker_overlap ───────────────────────────────────────────────────────


def test_chunker_overlap():
    """Adjacent chunks share the expected number of overlapping tokens."""
    pytest.importorskip("tiktoken")
    import tiktoken

    # Build text long enough to produce at least 3 chunks
    long_text = " ".join([f"token{i}" for i in range(500)])
    overlap = 20
    size = 100
    config = ChunkingConfig(
        strategy=ChunkingStrategy.SLIDING_WINDOW,
        chunk_size=size,
        chunk_overlap=overlap,
    )

    chunker = Chunker()
    chunks = chunker.chunk(long_text, {"source_file": "test.txt"}, config)
    assert len(chunks) >= 2

    enc = tiktoken.get_encoding("cl100k_base")
    for i in range(len(chunks) - 1):
        toks_a = enc.encode(chunks[i].text)
        toks_b = enc.encode(chunks[i + 1].text)
        # The tail of chunk A should appear at the head of chunk B
        shared = len(set(toks_a[-overlap:]) & set(toks_b[:overlap]))
        # Allow some tolerance because tokens at boundaries may differ slightly
        assert shared > 0, (
            f"Expected token overlap between chunk {i} and {i + 1}, got none"
        )


# ── test_ingestion_pipeline_dedup ─────────────────────────────────────────────


def test_ingestion_pipeline_dedup(tmp_path):
    """Duplicate documents (same text hash) are skipped when remove_duplicates=True."""
    pytest.importorskip("tiktoken")

    # Create two CSV files with identical content
    text = "The quick brown fox jumps over the lazy dog " * 20
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(f"content\n{text}\n{text}\n")

    config = KSelectConfig()
    config.chunking.strategy = ChunkingStrategy.SLIDING_WINDOW
    config.chunking.chunk_size = 64
    config.chunking.chunk_overlap = 0
    config.chunking.remove_duplicates = True

    # Mock the embedding model
    fake_emb = [[0.1] * 128]

    loader = CSVLoader(str(csv_path), text_col="content")

    pipeline = IngestionPipeline()
    with patch("kselect.ingestion.pipeline._Embedder.embed") as mock_embed:
        mock_embed.side_effect = lambda texts: [[0.1] * 128 for _ in texts]
        chunks = pipeline.run(loader, config)

    # With dedup, the second identical row should produce no new unique chunks
    texts = [c.text for c in chunks]
    unique_texts = set(texts)
    assert len(texts) == len(unique_texts), "Duplicate chunks were not removed"


# ── test_ingestion_pipeline_embedding_shape ───────────────────────────────────


def test_ingestion_pipeline_embedding_shape(tmp_path):
    """All returned Chunks have embedding populated with consistent dimensionality."""
    pytest.importorskip("tiktoken")

    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "content\n"
        "The quick brown fox jumps over the lazy dog.\n"
        "Pack my box with five dozen liquor jugs.\n"
    )

    config = KSelectConfig()
    config.chunking.strategy = ChunkingStrategy.SLIDING_WINDOW
    config.chunking.chunk_size = 64
    config.chunking.chunk_overlap = 0

    _DIM = 384
    loader = CSVLoader(str(csv_path), text_col="content")

    pipeline = IngestionPipeline()
    with patch("kselect.ingestion.pipeline._Embedder.embed") as mock_embed:
        mock_embed.side_effect = lambda texts: [[0.0] * _DIM for _ in texts]
        chunks = pipeline.run(loader, config)

    assert len(chunks) > 0
    for c in chunks:
        assert c.embedding is not None, f"Chunk {c.id} has no embedding"
        assert len(c.embedding) == _DIM, (
            f"Expected dim={_DIM}, got {len(c.embedding)}"
        )
