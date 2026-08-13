"""One adapter for every OpenAI-compatible endpoint.

LM Studio, Ollama, Groq, OpenRouter, vLLM, llama.cpp's server and OpenAI
itself all speak the same `/v1/chat/completions` wire format. That means
running Anchor fully offline on your laptop and running it against a hosted
free tier are the same code path with a different base URL — which is why
switching is a .env change rather than a rewrite.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional, Sequence

import openai

from ..config import Settings
from ..prompts import build_messages, find_invalid_citations, system_prompt_for
from ..retrieval import RetrievedChunk
from .base import Event, Generator, empty_usage

logger = logging.getLogger(__name__)


class OpenAICompatGenerator(Generator):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.name = settings.llm_provider
        self._preset = settings.preset
        self._model: Optional[str] = settings.resolved_model or None
        self._client = openai.AsyncOpenAI(
            base_url=settings.resolved_base_url,
            # Local runtimes ignore this but the SDK rejects an empty string.
            api_key=settings.resolved_api_key or "not-needed",
            max_retries=1,
        )

    async def _resolve_model(self) -> str:
        """Ask a local runtime what it has loaded.

        LM Studio serves whatever model the user loaded in the GUI, and its
        id is unpredictable. Rather than make people copy it into .env, we
        read /v1/models and take the first one.
        """
        if self._model:
            return self._model
        listing = await self._client.models.list()
        models = [m.id for m in listing.data]
        if not models:
            raise RuntimeError(
                f"{self.name} is reachable but has no model loaded. "
                "Load one in the app, or set ANCHOR_LLM_MODEL."
            )
        self._model = models[0]
        logger.info("Auto-detected model from %s: %s", self.name, self._model)
        return self._model

    async def stream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[dict],
    ) -> AsyncIterator[Event]:
        try:
            model = await self._resolve_model()
        except Exception as exc:  # noqa: BLE001 - mapped to a stream event
            logger.warning("Model resolution failed: %s", exc)
            yield {"type": "error", "message": self._friendly(exc)}
            return

        messages = [
            {"role": "system", "content": system_prompt_for(chunks)},
            *build_messages(question, chunks, history),
        ]

        request = {
            "model": model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            # Low but non-zero: grounded extraction wants determinism, and 0
            # makes some local runtimes loop on repeated phrases.
            "temperature": self.settings.temperature,
            "stream": True,
        }
        if self._preset.usage_in_stream:
            request["stream_options"] = {"include_usage": True}

        answer: list[str] = []
        usage = empty_usage()
        stop_reason = "end_turn"
        started = False

        try:
            stream = await self._client.chat.completions.create(**request)
            yield {"type": "status", "stage": "writing"}

            async for chunk in stream:
                if chunk.usage is not None:
                    usage = {
                        "input_tokens": chunk.usage.prompt_tokens or 0,
                        "output_tokens": chunk.usage.completion_tokens or 0,
                        "cache_read_input_tokens": 0,
                    }
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason:
                    stop_reason = (
                        "max_tokens" if choice.finish_reason == "length" else "end_turn"
                    )
                text = choice.delta.content if choice.delta else None
                if text:
                    started = True
                    answer.append(text)
                    yield {"type": "token", "text": text}
        except Exception as exc:  # noqa: BLE001 - mapped to a stream event
            logger.warning("%s generation failed: %s", self.name, exc)
            if not started:
                yield {"type": "error", "message": self._friendly(exc)}
                return
            # Partial answer already delivered; report the truncation rather
            # than discarding what the user can already read.
            yield {"type": "error", "message": f"Stream interrupted: {exc}"}

        joined = "".join(answer)
        yield {
            "type": "done",
            "model": model,
            "stop_reason": stop_reason,
            "usage": usage,
            "invalid_citations": find_invalid_citations(joined, len(chunks)),
        }

    def _friendly(self, exc: Exception) -> str:
        base = self.settings.resolved_base_url or "the provider"
        if isinstance(exc, openai.APIConnectionError):
            if self._preset.local:
                return (
                    f"Could not reach {self.name} at {base}. Start it and load "
                    f"a model — for LM Studio that's the Developer tab's "
                    f"'Start Server'; for Ollama, `ollama serve`."
                )
            return f"Could not reach {base}. Check your network connection."
        if isinstance(exc, openai.AuthenticationError):
            env_var = self._preset.api_key_env or "the provider's API key"
            return f"{self.name} rejected the credentials. Check {env_var}."
        if isinstance(exc, openai.NotFoundError):
            return (
                f"Model '{self._model}' is not available on {self.name}. "
                f"Set ANCHOR_LLM_MODEL to one that is."
            )
        if isinstance(exc, openai.RateLimitError):
            return f"Rate limited by {self.name}. Try again in a moment."
        return f"Generation failed: {exc}"
