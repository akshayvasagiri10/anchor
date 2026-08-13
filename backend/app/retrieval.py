"""Hybrid retrieval: BM25 (lexical) + cosine over dense vectors, fused with
Reciprocal Rank Fusion.

Why hybrid: the two strategies fail in opposite directions. BM25 nails exact
terms — product codes, error strings, proper nouns — and whiffs on paraphrase.
Dense vectors handle paraphrase and whiff on rare literals they never saw in
pretraining. Fusing them recovers both, and RRF does it without needing the
two score distributions to be comparable (they aren't: BM25 is an unbounded
negative log-ish score, cosine is [-1, 1]).

RRF:  score(d) = sum over strategies of  1 / (k + rank(d))
Rank-based, so no normalization, no tuning of a mixing weight. k=60 is the
constant from Cormack et al. (2009) and is a genuinely good default.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from . import embeddings as emb

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: str
    document_title: str
    source: str
    ordinal: int
    text: str
    score: float
    lexical_rank: Optional[int] = None
    dense_rank: Optional[int] = None

    @property
    def matched_by(self) -> list[str]:
        out = []
        if self.lexical_rank is not None:
            out.append("keyword")
        if self.dense_rank is not None:
            out.append("semantic")
        return out


@dataclass
class VectorCache:
    """All chunk embeddings held as one (n, dim) matrix.

    Brute force is the right call at this scale: 50k chunks x 384 dims is a
    73MB float32 matrix, and a single query is one matmul — sub-millisecond.
    An ANN index would add a dependency and an accuracy loss to solve a
    problem this project does not have. If the corpus outgrows memory, this
    is the one class to swap.
    """

    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    dirty: bool = True

    def invalidate(self) -> None:
        self.dirty = True

    def refresh(self, conn: sqlite3.Connection, dim: int) -> None:
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL ORDER BY id"
        ).fetchall()
        if not rows:
            self.ids = np.zeros(0, dtype=np.int64)
            self.matrix = np.zeros((0, dim), dtype=np.float32)
        else:
            self.ids = np.fromiter((r["id"] for r in rows), dtype=np.int64, count=len(rows))
            self.matrix = emb.unpack_matrix([r["embedding"] for r in rows], dim)
        self.dirty = False

    def search(
        self, conn: sqlite3.Connection, query_vector: np.ndarray, dim: int, top_k: int
    ) -> list[tuple[int, float]]:
        if self.dirty:
            self.refresh(conn, dim)
        if self.matrix.shape[0] == 0:
            return []
        # Both sides are L2-normalized at write time, so the dot product *is*
        # the cosine similarity.
        scores = self.matrix @ query_vector.astype(np.float32)
        k = min(top_k, scores.shape[0])
        # argpartition is O(n) vs argsort's O(n log n); we only need the top k.
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(self.ids[i]), float(scores[i])) for i in top]


def build_fts_query(question: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every token is wrapped in double quotes so punctuation, operators, and
    reserved words in user input can never be interpreted as query syntax.
    """
    tokens = [t.lower() for t in _TOKEN.findall(question) if len(t) > 1]
    if not tokens:
        return ""
    # Dedupe while preserving order.
    seen: set[str] = set()
    unique = [t for t in tokens if not (t in seen or seen.add(t))]
    return " OR ".join('"%s"' % t for t in unique)


def lexical_search(
    conn: sqlite3.Connection, question: str, limit: int
) -> list[tuple[int, float]]:
    """BM25 via SQLite FTS5. Lower bm25() is better, so we negate it."""
    match = build_fts_query(question)
    if not match:
        return []
    try:
        rows = conn.execute(
            """
            SELECT rowid AS chunk_id, bm25(chunks_fts) AS score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        # Malformed MATCH despite quoting (shouldn't happen) — fail soft so
        # the dense half of the search still answers.
        return []
    return [(int(r["chunk_id"]), -float(r["score"])) for r in rows]


def dense_search(
    conn: sqlite3.Connection,
    cache: VectorCache,
    embedder: Optional[emb.Embedder],
    question: str,
    limit: int,
) -> list[tuple[int, float]]:
    if embedder is None:
        return []
    query_vector = embedder.encode_one(question)
    return cache.search(conn, query_vector, embedder.dim, limit)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[int, float]]], *, k: int = 60
) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _score) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def hydrate(
    conn: sqlite3.Connection, chunk_ids: Sequence[int]
) -> dict[int, sqlite3.Row]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    rows = conn.execute(
        f"""
        SELECT c.id, c.document_id, c.ordinal, c.text,
               d.title AS document_title, d.source
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id IN ({placeholders})
        """,
        tuple(chunk_ids),
    ).fetchall()
    return {int(r["id"]): r for r in rows}


def search(
    conn: sqlite3.Connection,
    cache: VectorCache,
    embedder: Optional[emb.Embedder],
    question: str,
    *,
    top_k: int = 6,
    candidate_pool: int = 30,
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """Run both strategies, fuse, hydrate, and return the top_k chunks."""
    lexical = lexical_search(conn, question, candidate_pool)
    dense = dense_search(conn, cache, embedder, question, candidate_pool)

    if not lexical and not dense:
        return []

    fused = reciprocal_rank_fusion([lexical, dense], k=rrf_k)
    lexical_ranks = {cid: i for i, (cid, _) in enumerate(lexical, start=1)}
    dense_ranks = {cid: i for i, (cid, _) in enumerate(dense, start=1)}

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    rows = hydrate(conn, [cid for cid, _ in ordered])

    results: list[RetrievedChunk] = []
    for chunk_id, score in ordered:
        row = rows.get(chunk_id)
        if row is None:  # deleted between search and hydrate
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=row["document_id"],
                document_title=row["document_title"],
                source=row["source"],
                ordinal=int(row["ordinal"]),
                text=row["text"],
                score=score,
                lexical_rank=lexical_ranks.get(chunk_id),
                dense_rank=dense_ranks.get(chunk_id),
            )
        )
    return results
