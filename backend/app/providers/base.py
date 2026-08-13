"""The contract every generation backend implements."""

from __future__ import annotations

from typing import AsyncIterator, Dict, Sequence

from ..retrieval import RetrievedChunk

# Event shapes yielded by `Generator.stream`, serialized straight to SSE:
#   {"type": "status", "stage": "thinking" | "writing"}
#   {"type": "token",  "text": str}
#   {"type": "done",   "model": str, "stop_reason": str, "usage": {...},
#                      "invalid_citations": [int, ...]}
#   {"type": "error",  "message": str}
Event = Dict[str, object]


class Generator:
    """Base class rather than a Protocol so providers share the error mapping.

    Subclasses implement `_stream` and inherit uniform failure handling: no
    matter which backend breaks, the client sees one `error` event and the
    stream closes cleanly instead of hanging.
    """

    name: str = "base"

    async def stream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict],
    ) -> AsyncIterator[Event]:
        raise NotImplementedError


def empty_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}
