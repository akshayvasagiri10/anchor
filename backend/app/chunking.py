"""Sentence-aware chunking with overlap.

Naive fixed-width slicing cuts mid-sentence, which measurably hurts both
lexical and dense retrieval: the embedding of half a sentence is noise, and a
BM25 hit on a truncated clause gives the model an unusable snippet. So we
segment on sentence boundaries first, then pack sentences into chunks up to
the target size, carrying a tail of the previous chunk forward as overlap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Split after ., !, ?, or a blank line — but not after a common abbreviation
# or a decimal point. Good enough without dragging in an NLP dependency.
#
# The lookbehinds sit at the split position, i.e. *after* the period, so each
# one must include that period: `(?<!\bDr\.)`, not `(?<!\bDr)`.
_ABBREVIATIONS = (
    r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bDr\.)(?<!\bSt\.)"
    r"(?<!\bNo\.)(?<!\be\.g\.)(?<!\bi\.e\.)"
)
_SENTENCE_END = re.compile(
    r"(?<=[.!?])" + _ABBREVIATIONS + r"(?<!\d\.)\s+|\n{2,}",
)
_WHITESPACE = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    text: str

    @property
    def n_chars(self) -> int:
        return len(self.text)


def normalize(text: str) -> str:
    """Collapse horizontal whitespace and stray carriage returns.

    Deliberately preserves blank lines: they are the strongest paragraph
    signal we have, and the splitter uses them.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_END.split(text)]
    return [p for p in parts if p]


def _hard_split(sentence: str, size: int) -> list[str]:
    """A single 'sentence' longer than a whole chunk (minified JSON, a table,
    a wall of text with no punctuation). Break it on word boundaries."""
    out: list[str] = []
    buf: list[str] = []
    length = 0
    for word in sentence.split(" "):
        if len(word) > size:
            # A single token wider than a whole chunk — a base64 blob, a long
            # hash, a minified line. There is no boundary to respect, so slice
            # it. Without this the packer emits an oversized chunk.
            if buf:
                out.append(" ".join(buf))
                buf, length = [], 0
            out.extend(word[i : i + size] for i in range(0, len(word), size))
            continue
        # +1 for the space we'll rejoin with.
        if length + len(word) + 1 > size and buf:
            out.append(" ".join(buf))
            buf, length = [], 0
        buf.append(word)
        length += len(word) + 1
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_text(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 120,
) -> list[Chunk]:
    """Pack sentences into overlapping chunks.

    Size invariant: every returned chunk is at most ``chunk_size + overlap``
    characters. The slack exists because a chunk is *carried overlap* plus
    *fresh content*, and the packer only checks the budget before adding the
    next sentence — so one sentence may push past ``chunk_size`` before the
    flush happens.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    # Guard against degenerate configs where the minimum is a large fraction
    # of the chunk itself, which would fold nearly every tail backwards.
    min_chunk_size = min(min_chunk_size, chunk_size // 2)
    max_chunk_size = chunk_size + overlap

    text = normalize(text)
    if not text:
        return []

    sentences: list[str] = []
    for sentence in split_sentences(text):
        if len(sentence) > chunk_size:
            sentences.extend(_hard_split(sentence, chunk_size))
        else:
            sentences.append(sentence)

    chunks: list[str] = []
    buf: list[str] = []
    length = 0

    for sentence in sentences:
        if length + len(sentence) + 1 > chunk_size and buf:
            chunks.append(" ".join(buf))
            # Carry the tail of this chunk into the next one so a fact that
            # straddles a boundary survives in at least one chunk intact.
            buf, length = _overlap_tail(buf, overlap)
        buf.append(sentence)
        length += len(sentence) + 1

    if buf:
        tail = " ".join(buf)
        # Don't emit a runt final chunk — a 20-character chunk is noise in
        # both indexes. Fold it backwards, but only when doing so keeps the
        # size invariant; otherwise a short tail would silently produce an
        # oversized chunk.
        folded = len(chunks[-1]) + 1 + len(tail) if chunks else 0
        if chunks and len(tail) < min_chunk_size and folded <= max_chunk_size:
            chunks[-1] = (chunks[-1] + " " + tail).strip()
        else:
            chunks.append(tail)

    return [Chunk(ordinal=i, text=c.strip()) for i, c in enumerate(chunks) if c.strip()]


_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
MAX_BREADCRUMB = 120


@dataclass(frozen=True)
class Section:
    """A heading and the body text beneath it."""

    heading_path: list[str]
    body: str


def split_sections(text: str) -> list[Section]:
    """Split Markdown into sections, tracking the heading hierarchy.

    Headings are the strongest topical boundary a document gives you. A chunk
    that spans `## Refunds` into `## Shipping` dilutes both topics; splitting
    on headings first keeps each chunk about one thing.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [Section([], text)]

    sections: list[Section] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section([], preamble))

    stack: list[tuple[int, str]] = []
    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append(Section([t for _, t in stack], body))
    return sections


def chunk_document(
    text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 120,
) -> list[Chunk]:
    """Section-aware chunking — the entry point ingestion should call.

    Each chunk carries its heading breadcrumb as a prefix. That prefix does
    real work in both indexes: BM25 can now match the word "Refunds" against
    a chunk that never repeats it in the body, and the dense embedding gets
    topical context it would otherwise have to infer.
    """
    text = normalize(text)
    if not text:
        return []

    chunks: list[Chunk] = []
    ordinal = 0

    for section in split_sections(text):
        breadcrumb = " › ".join(section.heading_path)[:MAX_BREADCRUMB]
        prefix = f"{breadcrumb}\n\n" if breadcrumb else ""
        # Reserve room for the prefix so the final chunk still respects the
        # size budget. The floor keeps a pathological heading from starving
        # the body entirely.
        budget = max(chunk_size - len(prefix), overlap * 2)
        for part in chunk_text(
            section.body,
            chunk_size=budget,
            overlap=overlap,
            min_chunk_size=min_chunk_size,
        ):
            chunks.append(Chunk(ordinal=ordinal, text=prefix + part.text))
            ordinal += 1

    return chunks


def _overlap_tail(buf: list[str], overlap: int) -> tuple[list[str], int]:
    """Return the trailing sentences of ``buf`` that fit within ``overlap``."""
    tail: list[str] = []
    length = 0
    for sentence in reversed(buf):
        if length + len(sentence) + 1 > overlap:
            break
        tail.insert(0, sentence)
        length += len(sentence) + 1

    if not tail and buf:
        # The final unit is itself longer than the overlap window — typical of
        # hard-split text with no sentence boundaries. Without this branch
        # such content would get no overlap at all, leaving every seam
        # uncovered. Carry its trailing words instead.
        words = buf[-1].split(" ")
        carried: list[str] = []
        for word in reversed(words):
            if length + len(word) + 1 > overlap:
                break
            carried.insert(0, word)
            length += len(word) + 1
        if carried:
            tail = [" ".join(carried)]

    return tail, length
