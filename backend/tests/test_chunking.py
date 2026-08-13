from __future__ import annotations

from app.chunking import (
    chunk_document,
    chunk_text,
    normalize,
    split_sections,
    split_sentences,
)

MARKDOWN = """# Handbook

Intro paragraph before any subheading.

## Refunds

Customers may request a refund within 30 days of purchase.

### Digital goods

Digital goods are refundable only if undownloaded.

## Shipping

Standard shipping takes five business days.
"""


def test_sections_track_the_heading_hierarchy():
    sections = split_sections(normalize(MARKDOWN))
    paths = [s.heading_path for s in sections]
    assert ["Handbook"] in paths
    assert ["Handbook", "Refunds"] in paths
    assert ["Handbook", "Refunds", "Digital goods"] in paths
    # A sibling h2 must pop the h3 off the stack, not nest under it.
    assert ["Handbook", "Shipping"] in paths


def test_document_without_headings_is_a_single_section():
    sections = split_sections("Just prose. No headings at all.")
    assert len(sections) == 1
    assert sections[0].heading_path == []


def test_chunks_carry_their_heading_breadcrumb():
    chunks = chunk_document(MARKDOWN, chunk_size=300, overlap=60)
    refund_chunk = next(c for c in chunks if "30 days" in c.text)
    assert refund_chunk.text.startswith("Handbook › Refunds")
    # The breadcrumb is what makes BM25 match "refund" against a body that
    # never repeats the word.
    assert "Refunds" in refund_chunk.text


def test_headings_split_topics_apart():
    chunks = chunk_document(MARKDOWN, chunk_size=1200, overlap=200)
    # Everything would fit in one 1200-char chunk without heading awareness.
    assert len(chunks) >= 4
    assert not any("30 days" in c.text and "five business days" in c.text for c in chunks)


def test_chunk_document_respects_the_size_invariant_with_prefixes():
    body = "Sentence with several words in it. " * 200
    text = f"# Title\n\n## Section A\n\n{body}\n\n## Section B\n\n{body}"
    for chunk_size, overlap in [(300, 60), (1200, 200)]:
        chunks = chunk_document(text, chunk_size=chunk_size, overlap=overlap)
        assert chunks
        assert all(c.n_chars <= chunk_size + overlap for c in chunks)


def test_chunk_document_ordinals_are_contiguous_across_sections():
    chunks = chunk_document(MARKDOWN, chunk_size=200, overlap=40)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_normalize_collapses_horizontal_whitespace_but_keeps_paragraphs():
    text = "Hello    world.\r\n\r\n\r\nNext   para."
    assert normalize(text) == "Hello world.\n\nNext para."


def test_split_sentences_ignores_common_abbreviations():
    sentences = split_sentences("Dr. Rao met Mr. Iyer. They agreed.")
    assert sentences == ["Dr. Rao met Mr. Iyer.", "They agreed."]


def test_split_sentences_ignores_decimals():
    assert split_sentences("Version 3.9 shipped. It was fine.") == [
        "Version 3.9 shipped.",
        "It was fine.",
    ]


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert chunks
    assert all(c.n_chars <= 300 + 60 for c in chunks)


def test_chunks_overlap_so_boundary_facts_survive():
    text = (
        "Alpha is the first letter. Beta is the second letter. "
        "Gamma is the third letter. Delta is the fourth letter. "
        "Epsilon is the fifth letter. Zeta is the sixth letter."
    )
    chunks = chunk_text(text, chunk_size=120, overlap=60)
    assert len(chunks) > 1
    # Consecutive chunks must share text, otherwise a fact spanning the seam
    # exists in neither chunk in full.
    for a, b in zip(chunks, chunks[1:]):
        assert any(word in b.text for word in a.text.split()[-4:])


def test_ordinals_are_contiguous_from_zero():
    chunks = chunk_text("One. Two. Three. Four. Five." * 40, chunk_size=200, overlap=40)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_sentence_longer_than_a_chunk_is_hard_split():
    monster = "word " * 500  # one 2500-char "sentence", no punctuation
    chunks = chunk_text(monster, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(c.n_chars <= 360 for c in chunks)


def test_runt_tail_is_folded_into_previous_chunk():
    # Two sentences fill a chunk exactly; the third would start a new one but
    # is far too short to stand alone, so it should merge backwards.
    text = "{} {} Tiny.".format("A" * 144 + ".", "B" * 147 + ".")
    chunks = chunk_text(text, chunk_size=300, overlap=60, min_chunk_size=40)
    assert len(chunks) == 1
    assert chunks[-1].text.endswith("Tiny.")


def test_size_invariant_holds_across_shapes():
    """Every chunk must be <= chunk_size + overlap, whatever the input looks
    like. This is the invariant the retrieval layer and the token budget both
    depend on."""
    shapes = [
        "Short. " * 300,
        "word " * 900,  # no sentence boundaries at all
        "Para one.\n\nPara two.\n\n" * 80,
        ("A" * 500 + ". ") * 6,  # sentences far longer than a chunk
        "Mixed. " + "x" * 700 + ". Then normal sentences follow here. " * 20,
    ]
    for chunk_size, overlap in [(300, 60), (1200, 200), (150, 20)]:
        for shape in shapes:
            chunks = chunk_text(shape, chunk_size=chunk_size, overlap=overlap)
            oversized = [c for c in chunks if c.n_chars > chunk_size + overlap]
            assert not oversized, (
                f"chunk_size={chunk_size} overlap={overlap}: "
                f"{[c.n_chars for c in oversized]} exceed "
                f"{chunk_size + overlap}"
            )


def test_hard_split_content_still_gets_overlap():
    """Text with no sentence boundaries must not lose seam coverage."""
    chunks = chunk_text("word " * 400, chunk_size=300, overlap=60)
    assert len(chunks) > 2
    # Each chunk after the first starts with words carried from its
    # predecessor, so the total length exceeds the source length.
    assert sum(c.n_chars for c in chunks) > len("word " * 400)


def test_empty_input_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_overlap_must_be_smaller_than_chunk_size():
    try:
        chunk_text("hello", chunk_size=100, overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
