"""SQLite storage: documents, chunks, float32 embedding blobs, and an FTS5
index that gives us Okapi BM25 for free via SQLite's built-in `bm25()`.

Using external-content FTS5 (`content='chunks'`) means the text is stored once
and the index stays in sync through triggers — no double writes, no drift.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    title         TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL UNIQUE,
    n_chunks      INTEGER NOT NULL DEFAULT 0,
    n_chars       INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    n_chars     INTEGER NOT NULL,
    embedding   BLOB,
    UNIQUE (document_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Keep the FTS index in lockstep with the chunks table.
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with sane defaults and the schema applied."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Explicit transaction — sqlite3's autocommit default is a footgun for
    multi-statement ingests where a partial write is worse than no write."""
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


def assert_fts5(conn: sqlite3.Connection) -> None:
    """Fail loudly at startup rather than silently losing lexical retrieval."""
    row = conn.execute(
        "SELECT 1 FROM pragma_compile_options WHERE compile_options LIKE 'ENABLE_FTS5%'"
    ).fetchone()
    if row is None:
        # Some builds don't report it via pragma; probe directly.
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_probe USING fts5(x)")
            conn.execute("DROP TABLE IF EXISTS _fts_probe")
        except sqlite3.OperationalError as exc:  # pragma: no cover
            raise RuntimeError(
                "This SQLite build lacks FTS5, which Anchor needs for lexical "
                "retrieval. Install a Python linked against a newer SQLite."
            ) from exc
