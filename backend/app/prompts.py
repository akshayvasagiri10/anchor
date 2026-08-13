"""Prompt construction and citation checking, shared by every provider.

The rules are written as a short numbered list rather than prose. That is a
deliberate concession to small local models: a 4B model follows six terse
imperatives far more reliably than it follows a nuanced paragraph, and the
larger models lose nothing from the same phrasing.
"""

from __future__ import annotations

import re
from typing import Sequence

from .retrieval import RetrievedChunk

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources given to you.

Rules:
1. Use only the sources. Never use outside knowledge. Never guess.
2. End every factual sentence with its source number in brackets, like [2]. Use [1][3] when a sentence draws on two sources.
3. Only cite numbers that actually appear in the sources. Never invent a number.
4. If the sources do not answer the question, reply exactly: "The sources don't cover that." Then stop.
5. If the sources only partly answer it, answer that part and say what is missing.
6. If two sources disagree, say so and cite both.

Style: answer first, detail after. Be brief. Write plain prose, not bullet lists, unless the question asks for a list. Never refer to "the sources" or "the context" in your wording — just cite."""

NO_CONTEXT_PROMPT = """No documents matched the user's question.

Reply in one sentence that you have nothing on file about it. Do not answer from your own knowledge. Do not apologise at length."""

_CITATION = re.compile(r"\[(\d+)\]")


def format_sources(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered block for the user turn.

    Sources go in the user turn rather than the system prompt on purpose: the
    system prompt is identical across every request, so it stays a stable
    cache prefix while the volatile per-query context sits after it.
    """
    return "\n\n".join(
        f'<source id="{i}" title="{chunk.document_title}">\n'
        f"{chunk.text}\n"
        f"</source>"
        for i, chunk in enumerate(chunks, start=1)
    )


def build_messages(
    question: str,
    chunks: Sequence[RetrievedChunk],
    history: Sequence[dict],
) -> list[dict]:
    """Prior turns first, then the current question with its fresh sources.

    Only the current turn carries sources. Re-attaching context from earlier
    turns would balloon the prompt and let stale excerpts outvote the ones
    actually retrieved for the question being asked.
    """
    messages: list[dict] = []
    for turn in history:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    if chunks:
        user_content = f"{format_sources(chunks)}\n\n---\n\nQuestion: {question}"
    else:
        user_content = f"Question: {question}"

    messages.append({"role": "user", "content": user_content})
    return messages


def system_prompt_for(chunks: Sequence[RetrievedChunk]) -> str:
    return SYSTEM_PROMPT if chunks else NO_CONTEXT_PROMPT


def find_invalid_citations(answer: str, n_sources: int) -> list[int]:
    """Citation numbers the model used that do not exist.

    This is the project's own honesty check, and it earns its keep the moment
    you point Anchor at a small local model: a 4B model will cheerfully write
    [4] when it was handed three sources. Surfacing that is the entire premise
    of the project — a citation nobody verifies is just a confident guess.
    """
    seen = {int(m.group(1)) for m in _CITATION.finditer(answer)}
    return sorted(n for n in seen if n < 1 or n > n_sources)
