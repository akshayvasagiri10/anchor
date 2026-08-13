"""Provider selection.

`ANCHOR_LLM_PROVIDER` picks the backend. Everything except `anthropic` is
OpenAI-compatible and shares one adapter, so adding vLLM, llama.cpp or a new
hosted free tier is a row in `PRESETS`, not a new module.
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings
from .base import Generator

_generator: Optional[Generator] = None
_generator_key: Optional[tuple] = None


def build_generator(settings: Settings) -> Generator:
    if settings.llm_provider == "anthropic":
        from .anthropic_provider import AnthropicGenerator

        return AnthropicGenerator(settings)

    # Validates the provider name and raises a helpful error for typos.
    settings.preset
    from .openai_compat import OpenAICompatGenerator

    return OpenAICompatGenerator(settings)


def get_generator(settings: Settings) -> Generator:
    """One client per configuration — they pool HTTP connections internally.

    Keyed on the settings that define the connection so a test (or a config
    reload) that switches provider does not silently keep the old client.
    """
    global _generator, _generator_key
    key = (
        settings.llm_provider,
        settings.resolved_base_url,
        settings.resolved_model,
    )
    if _generator is None or _generator_key != key:
        _generator = build_generator(settings)
        _generator_key = key
    return _generator


def reset() -> None:
    """Test hook — drops the cached client."""
    global _generator, _generator_key
    _generator = None
    _generator_key = None


__all__ = ["Generator", "build_generator", "get_generator", "reset"]
