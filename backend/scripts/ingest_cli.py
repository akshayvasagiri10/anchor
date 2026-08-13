#!/usr/bin/env python3
"""Bulk-ingest files or directories from the command line.

    python scripts/ingest_cli.py ../data/samples
    python scripts/ingest_cli.py handbook.pdf notes.md --no-embed
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings as emb  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import assert_fts5, connect  # noqa: E402
from app.ingest import SUPPORTED_SUFFIXES, UnsupportedFileType, ingest_file  # noqa: E402
from app.retrieval import VectorCache  # noqa: E402


def collect(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_dir():
            files.extend(
                sorted(
                    p
                    for p in path.rglob("*")
                    if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                )
            )
        elif path.is_file():
            files.append(path)
        else:
            print(f"  skip  {path} (not found)", file=sys.stderr)
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Anchor.")
    parser.add_argument("paths", nargs="+", help="Files or directories to ingest")
    parser.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip dense embeddings (BM25-only retrieval, much faster)",
    )
    args = parser.parse_args()

    settings = get_settings()
    conn = connect(settings.db_path)
    assert_fts5(conn)
    cache = VectorCache()

    embedder = None
    if not args.no_embed:
        embedder = emb.get_embedder(
            settings.embedding_model, enabled=settings.embeddings_enabled
        )
        if embedder is None:
            print("! sentence-transformers unavailable — ingesting BM25-only\n")

    files = collect(args.paths)
    if not files:
        print("Nothing to ingest.", file=sys.stderr)
        return 1

    started = time.time()
    total_chunks = 0
    failures = 0

    for path in files:
        try:
            result = ingest_file(conn, cache, settings, path, embedder=embedder)
        except (UnsupportedFileType, ValueError) as exc:
            print(f"  fail  {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if result.skipped:
            print(f"  same  {path.name} (already ingested, unchanged)")
            continue

        verb = "recut" if result.replaced else "added"
        total_chunks += result.n_chunks
        print(
            f"  {verb} {path.name} -> {result.n_chunks} chunks, "
            f"{result.n_chars:,} chars"
            + ("" if result.embedded else "  [no embeddings]")
        )

    elapsed = time.time() - started
    print(
        f"\n{len(files) - failures}/{len(files)} files, "
        f"{total_chunks} new chunks in {elapsed:.1f}s"
    )
    conn.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
