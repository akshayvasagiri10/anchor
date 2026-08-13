"""Document loading and ingestion.

Ingest is idempotent by content hash: re-uploading an unchanged file is a
no-op, and re-uploading a *changed* file replaces the old version's chunks
rather than leaving stale ones to be retrieved alongside the new ones.
"""

from __future__ import annotations

import hashlib
import io
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import embeddings as emb
from .chunking import chunk_document
from .config import Settings
from .db import transaction
from .retrieval import VectorCache

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".rst", ".csv", ".json"}


class UnsupportedFileType(ValueError):
    pass


@dataclass
class IngestResult:
    document_id: str
    title: str
    n_chunks: int
    n_chars: int
    embedded: bool
    replaced: bool
    skipped: bool = False


def extract_text(filename: str, data: bytes) -> str:
    """Bytes -> plain text, dispatched on file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(data)
    if suffix in SUPPORTED_SUFFIXES:
        return data.decode("utf-8", errors="replace")
    raise UnsupportedFileType(
        f"Unsupported file type '{suffix or filename}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
    )


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one bad page shouldn't kill the doc
            logger.warning("Failed to extract page %d: %s", i, exc)
            text = ""
        if text.strip():
            # Page markers survive chunking and make citations traceable back
            # to a physical page.
            pages.append(f"[page {i}]\n{text.strip()}")
    return "\n\n".join(pages)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_text(
    conn: sqlite3.Connection,
    cache: VectorCache,
    settings: Settings,
    *,
    title: str,
    source: str,
    text: str,
    embedder: Optional[emb.Embedder] = None,
) -> IngestResult:
    """Chunk, embed, and store a document. Returns what actually happened."""
    digest = content_hash(text)

    existing = conn.execute(
        "SELECT id, title FROM documents WHERE content_hash = ?", (digest,)
    ).fetchone()
    if existing is not None:
        row = conn.execute(
            "SELECT n_chunks, n_chars FROM documents WHERE id = ?", (existing["id"],)
        ).fetchone()
        return IngestResult(
            document_id=existing["id"],
            title=existing["title"],
            n_chunks=int(row["n_chunks"]),
            n_chars=int(row["n_chars"]),
            embedded=False,
            replaced=False,
            skipped=True,
        )

    chunks = chunk_document(
        text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        min_chunk_size=settings.min_chunk_size,
    )
    if not chunks:
        raise ValueError("Document produced no text after extraction")

    vectors = None
    if embedder is not None:
        vectors = embedder.encode([c.text for c in chunks])

    # Same source path re-uploaded with different content: supersede it.
    prior = conn.execute(
        "SELECT id FROM documents WHERE source = ?", (source,)
    ).fetchone()

    document_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    total_chars = sum(c.n_chars for c in chunks)

    with transaction(conn):
        if prior is not None:
            conn.execute("DELETE FROM documents WHERE id = ?", (prior["id"],))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (prior["id"],))
        conn.execute(
            """
            INSERT INTO documents (id, title, source, content_hash, n_chunks, n_chars, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, title, source, digest, len(chunks), total_chars, now),
        )
        conn.executemany(
            """
            INSERT INTO chunks (document_id, ordinal, text, n_chars, embedding)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    document_id,
                    c.ordinal,
                    c.text,
                    c.n_chars,
                    emb.pack(vectors[i]) if vectors is not None else None,
                )
                for i, c in enumerate(chunks)
            ],
        )

    cache.invalidate()
    return IngestResult(
        document_id=document_id,
        title=title,
        n_chunks=len(chunks),
        n_chars=total_chars,
        embedded=vectors is not None,
        replaced=prior is not None,
    )


def ingest_file(
    conn: sqlite3.Connection,
    cache: VectorCache,
    settings: Settings,
    path: Path,
    *,
    embedder: Optional[emb.Embedder] = None,
) -> IngestResult:
    data = path.read_bytes()
    text = extract_text(path.name, data)
    return ingest_text(
        conn,
        cache,
        settings,
        title=path.stem.replace("_", " ").replace("-", " ").strip() or path.name,
        source=str(path),
        text=text,
        embedder=embedder,
    )


def delete_document(
    conn: sqlite3.Connection, cache: VectorCache, document_id: str
) -> bool:
    with transaction(conn):
        cur = conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        deleted = conn.execute(
            "DELETE FROM documents WHERE id = ?", (document_id,)
        ).rowcount
    if deleted or cur.rowcount:
        cache.invalidate()
    return bool(deleted)
