from __future__ import annotations

import numpy as np

from app import embeddings as emb
from app.ingest import ingest_text
from app.retrieval import (
    build_fts_query,
    lexical_search,
    reciprocal_rank_fusion,
    search,
)


def _seed(conn, cache, settings):
    ingest_text(
        conn,
        cache,
        settings,
        title="Refund Policy",
        source="refunds.md",
        text=(
            "Customers may request a refund within 30 days of purchase. "
            "Refunds are issued to the original payment method. "
            "Digital goods carry error code ERR_4417 when a refund fails."
        ),
    )
    ingest_text(
        conn,
        cache,
        settings,
        title="Shipping Guide",
        source="shipping.md",
        text=(
            "Standard shipping takes five to seven business days. "
            "Express shipping arrives the next business day. "
            "We do not ship to post office boxes."
        ),
    )


def test_fts_query_quotes_every_token():
    assert build_fts_query("refund policy") == '"refund" OR "policy"'


def test_fts_query_neutralizes_operator_injection():
    # Bare FTS5 would choke on these; quoting makes them literal terms.
    query = build_fts_query('NEAR("a" "b") AND * OR ^x')
    assert "*" not in query
    assert query.startswith('"near"')


def test_fts_query_dedupes_and_drops_single_chars():
    assert build_fts_query("a the the cat") == '"the" OR "cat"'


def test_fts_query_of_pure_punctuation_is_empty():
    assert build_fts_query("?!  ... ") == ""


def test_lexical_search_finds_exact_rare_token(conn, cache, settings):
    _seed(conn, cache, settings)
    hits = lexical_search(conn, "ERR_4417", limit=5)
    assert hits
    top_id = hits[0][0]
    text = conn.execute("SELECT text FROM chunks WHERE id = ?", (top_id,)).fetchone()
    assert "ERR_4417" in text["text"]


def test_lexical_search_on_unmatchable_query_returns_nothing(conn, cache, settings):
    _seed(conn, cache, settings)
    assert lexical_search(conn, "quokka platypus", limit=5) == []


def test_rrf_rewards_agreement_between_strategies():
    lexical = [(1, 9.0), (2, 8.0), (3, 7.0)]
    dense = [(3, 0.9), (1, 0.8), (9, 0.7)]
    fused = reciprocal_rank_fusion([lexical, dense], k=60)
    # 1 and 3 appear in both lists; 2 and 9 appear in one each.
    assert fused[1] > fused[2]
    assert fused[3] > fused[9]
    assert set(fused) == {1, 2, 3, 9}


def test_rrf_is_bounded_by_number_of_strategies():
    fused = reciprocal_rank_fusion([[(1, 1.0)], [(1, 1.0)]], k=60)
    assert fused[1] == 2 / 61


def test_search_falls_back_to_bm25_without_an_embedder(conn, cache, settings):
    _seed(conn, cache, settings)
    results = search(conn, cache, None, "refund within 30 days", top_k=3)
    assert results
    assert results[0].matched_by == ["keyword"]
    assert "refund" in results[0].text.lower()


def test_search_on_empty_corpus_returns_nothing(conn, cache, settings):
    assert search(conn, cache, None, "anything at all", top_k=5) == []


def test_deleting_a_document_removes_it_from_the_lexical_index(
    conn, cache, settings
):
    _seed(conn, cache, settings)
    doc = conn.execute("SELECT id FROM documents WHERE source='refunds.md'").fetchone()
    conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc["id"],))
    conn.commit()
    assert lexical_search(conn, "ERR_4417", limit=5) == []


class _StubEmbedder:
    """Deterministic fake so dense retrieval is testable without torch."""

    dim = 4

    def __init__(self, mapping):
        self.mapping = mapping

    def _vec(self, text: str) -> np.ndarray:
        for needle, vec in self.mapping.items():
            if needle in text.lower():
                v = np.array(vec, dtype=np.float32)
                return v / np.linalg.norm(v)
        v = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return v

    def encode(self, texts, batch_size=32):
        return np.stack([self._vec(t) for t in texts]).astype(np.float32)

    def encode_one(self, text):
        return self._vec(text)


def test_dense_retrieval_surfaces_a_paraphrase_bm25_would_miss(
    conn, cache, settings
):
    embedder = _StubEmbedder(
        {"refund": [1, 0, 0, 0], "shipping": [0, 1, 0, 0], "money back": [1, 0, 0, 0]}
    )
    ingest_text(
        conn,
        cache,
        settings,
        title="Refund Policy",
        source="refunds.md",
        text="Customers may request a refund within 30 days of purchase.",
        embedder=embedder,
    )
    ingest_text(
        conn,
        cache,
        settings,
        title="Shipping",
        source="shipping.md",
        text="Standard shipping takes five to seven business days.",
        embedder=embedder,
    )

    # "money back" shares no tokens with the refund chunk, so BM25 alone fails.
    assert lexical_search(conn, "money back", limit=5) == []

    results = search(conn, cache, embedder, "money back", top_k=2)
    assert results
    assert "refund" in results[0].text.lower()
    assert "semantic" in results[0].matched_by


def test_vector_cache_refreshes_after_ingest(conn, cache, settings):
    embedder = _StubEmbedder({"alpha": [1, 0, 0, 0]})
    ingest_text(
        conn, cache, settings, title="A", source="a.md",
        text="Alpha is the first letter of the Greek alphabet.",
        embedder=embedder,
    )
    assert search(conn, cache, embedder, "alpha", top_k=1)

    ingest_text(
        conn, cache, settings, title="B", source="b.md",
        text="Beta is the second letter of the Greek alphabet.",
        embedder=embedder,
    )
    # The cache was invalidated by the second ingest, so both are searchable.
    cache.refresh(conn, embedder.dim)
    assert cache.matrix.shape[0] == 2


def test_pack_unpack_roundtrip_preserves_vectors():
    v = np.array([0.1, -0.2, 0.3, 0.4], dtype=np.float32)
    v = v / np.linalg.norm(v)
    assert np.allclose(emb.unpack(emb.pack(v), 4), v, atol=1e-6)
