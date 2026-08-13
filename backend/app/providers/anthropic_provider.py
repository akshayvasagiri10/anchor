"""Claude via the official Anthropic SDK.

Kept as a first-class provider rather than the only one: it is the quality
ceiling to measure the local models against, and the grounding discipline
Anchor is built around is exactly where the gap shows up.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Sequence

import anthropic

from ..config import Settings
from ..prompts import build_messages, find_invalid_citations, system_prompt_for
from ..retrieval import RetrievedChunk
from .base import Event, Generator

logger = logging.getLogger(__name__)

MISSING_CREDENTIALS = (
    "No Anthropic credentials found. Set ANTHROPIC_API_KEY in your environment "
    "or in the project's .env file, then restart the server."
)


def _is_fallback_rejection(exc: Exception) -> bool:
    """Did this request fail *because* of the beta `fallbacks` parameter?

    An SDK too old to know the kwarg raises TypeError; an API that hasn't
    enabled the beta raises BadRequestError. Both name it in the message, so
    match on that rather than on the exception class — otherwise unrelated
    TypeErrors (a missing API key, for one) get misread as a beta problem.
    """
    return isinstance(exc, (TypeError, anthropic.BadRequestError)) and (
        "fallback" in str(exc).lower()
    )


def _friendly_error(exc: Exception) -> str:
    # Raised at request time, before any HTTP call, when the SDK cannot
    # resolve a credential from the environment.
    if isinstance(exc, TypeError) and "authentication method" in str(exc):
        return MISSING_CREDENTIALS
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "Anthropic rejected the credentials. Check that ANTHROPIC_API_KEY "
            "is a valid, active key."
        )
    if isinstance(exc, anthropic.RateLimitError):
        return "Rate limited by the Anthropic API. Try again in a moment."
    if isinstance(exc, anthropic.APIConnectionError):
        return "Could not reach the Anthropic API. Check your network connection."
    return f"Generation failed: {exc}"


class AnthropicGenerator(Generator):
    name = "anthropic"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = anthropic.AsyncAnthropic()
        # Server-side refusal fallback is a beta parameter. If this API key or
        # SDK build rejects it we downgrade once and remember, rather than
        # failing every subsequent request.
        self._use_fallback = True

    async def stream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict],
    ) -> AsyncIterator[Event]:
        params = {
            "model": self.settings.resolved_model,
            "max_tokens": self.settings.max_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt_for(chunks),
                    # The system prompt is byte-identical on every request, so
                    # it is worth caching. Sources sit after it in the user
                    # turn and never invalidate this prefix.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            # Chat latency is managed with effort, not by disabling thinking.
            "output_config": {"effort": self.settings.effort},
            "messages": build_messages(question, chunks, history),
        }

        # At most two attempts: the second only happens when the first was
        # rejected *because of* the beta fallback parameter, before any tokens
        # were produced. Retrying after partial output would duplicate text.
        for attempt in (1, 2):
            produced = False
            try:
                async for event in self._stream_once(params, len(chunks)):
                    produced = True
                    yield event
                return
            except Exception as exc:  # noqa: BLE001 - mapped to a stream event
                retryable = (
                    attempt == 1
                    and not produced
                    and self._use_fallback
                    and _is_fallback_rejection(exc)
                )
                if retryable:
                    logger.warning(
                        "Server-side refusal fallback unavailable, disabling it: %s",
                        exc,
                    )
                    self._use_fallback = False
                    continue
                logger.exception("Generation failed")
                yield {"type": "error", "message": _friendly_error(exc)}
                return

    async def _stream_once(self, params: dict, n_sources: int) -> AsyncIterator[Event]:
        if self._use_fallback:
            # Opus 5's safety classifiers can decline a request outright.
            # `fallbacks="default"` lets Anthropic re-serve it on a suitable
            # model inside the same call, routed by refusal category, instead
            # of handing the user a dead end.
            stream_cm = self._client.beta.messages.stream(
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                **params,
            )
        else:
            stream_cm = self._client.messages.stream(**params)

        answer: list[str] = []

        async with stream_cm as stream:
            emitted_text = False
            async for event in stream:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        # Thinking text is omitted by default, so the stream
                        # goes quiet here. Tell the UI to show a spinner
                        # instead of looking hung.
                        yield {"type": "status", "stage": "thinking"}
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta" and event.delta.text:
                        if not emitted_text:
                            emitted_text = True
                            yield {"type": "status", "stage": "writing"}
                        answer.append(event.delta.text)
                        yield {"type": "token", "text": event.delta.text}

            message = await stream.get_final_message()

        if message.stop_reason == "refusal":
            category = getattr(message.stop_details, "category", None)
            yield {
                "type": "error",
                "message": (
                    "This request was declined by the model's safety systems"
                    + (f" ({category})" if category else "")
                    + "."
                ),
            }
            return

        yield {
            "type": "done",
            "model": message.model,
            "stop_reason": message.stop_reason,
            "usage": {
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
                "cache_read_input_tokens": getattr(
                    message.usage, "cache_read_input_tokens", 0
                )
                or 0,
            },
            "invalid_citations": find_invalid_citations("".join(answer), n_sources),
        }
