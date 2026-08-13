"""Provider selection, the OpenAI-compatible adapter, and citation checking.

Every test here runs offline against a stubbed client — the point is the
adapter's contract (what it emits, how it fails), not any model's output.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai  # noqa: E402

from app import providers  # noqa: E402
from app.config import PRESETS, Settings  # noqa: E402
from app.prompts import find_invalid_citations  # noqa: E402
from app.providers.openai_compat import OpenAICompatGenerator  # noqa: E402
from app.retrieval import RetrievedChunk  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    providers.reset()
    yield
    providers.reset()


def _chunk(text: str = "Refunds are issued within 30 days.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        document_id="d",
        document_title="Policy",
        source="policy.md",
        ordinal=0,
        text=text,
        score=1.0,
    )


# --- Configuration --------------------------------------------------------


def test_default_provider_is_local_so_a_fresh_clone_needs_no_key():
    settings = Settings()
    assert settings.preset.local is True
    assert settings.credentials_present is True


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_resolves_without_error(name):
    settings = Settings(llm_provider=name)
    assert settings.resolved_model or settings.preset.local
    assert isinstance(settings.credentials_present, bool)


def test_unknown_provider_names_the_valid_options():
    with pytest.raises(ValueError) as excinfo:
        Settings(llm_provider="gpt5-turbo-max").preset
    message = str(excinfo.value)
    assert "gpt5-turbo-max" in message
    assert "lmstudio" in message and "ollama" in message


def test_explicit_settings_win_over_preset_defaults():
    settings = Settings(
        llm_provider="ollama",
        llm_model="gemma3:4b",
        llm_base_url="http://192.168.1.50:11434/v1",
    )
    assert settings.resolved_model == "gemma3:4b"
    assert settings.resolved_base_url == "http://192.168.1.50:11434/v1"


def test_hosted_provider_without_a_key_reports_missing_credentials(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert Settings(llm_provider="groq").credentials_present is False


def test_hosted_provider_reads_its_conventional_env_var(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    settings = Settings(llm_provider="groq")
    assert settings.resolved_api_key == "gsk-test"
    assert settings.credentials_present is True


def test_local_providers_supply_a_placeholder_key():
    # The OpenAI SDK rejects an empty api_key even when the server ignores it.
    assert Settings(llm_provider="lmstudio").resolved_api_key
    assert Settings(llm_provider="ollama").resolved_api_key


def test_factory_returns_the_openai_adapter_for_local_providers():
    generator = providers.build_generator(Settings(llm_provider="lmstudio"))
    assert isinstance(generator, OpenAICompatGenerator)


def test_generator_is_cached_but_rebuilt_when_the_provider_changes():
    first = providers.get_generator(Settings(llm_provider="lmstudio"))
    assert providers.get_generator(Settings(llm_provider="lmstudio")) is first
    second = providers.get_generator(Settings(llm_provider="ollama"))
    assert second is not first


# --- Citation checking ----------------------------------------------------


def test_valid_citations_are_not_flagged():
    assert find_invalid_citations("Within 30 days [1]. Also [2].", 3) == []


def test_out_of_range_citations_are_flagged():
    # This is the failure mode small local models actually exhibit.
    assert find_invalid_citations("Refunds take 30 days [4].", 3) == [4]


def test_zero_and_negative_style_citations_are_flagged():
    assert find_invalid_citations("See [0].", 3) == [0]


def test_citation_scan_ignores_ordinary_brackets():
    assert find_invalid_citations("An array [x] and a range [a-b].", 1) == []


# --- OpenAI-compatible adapter --------------------------------------------


def _delta_chunk(text=None, finish_reason=None, usage=None):
    """Mimic one `chat.completions` stream chunk."""
    choices = []
    if text is not None or finish_reason is not None:
        choices = [
            SimpleNamespace(
                delta=SimpleNamespace(content=text), finish_reason=finish_reason
            )
        ]
    return SimpleNamespace(choices=choices, usage=usage)


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        async def gen():
            for chunk in self._chunks:
                yield chunk

        return gen()


def _install_fake_client(generator, chunks=None, error=None, models=("local-model",)):
    """Replace the SDK client with a stub, capturing the request."""
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        if error is not None:
            raise error
        return _FakeStream(chunks or [])

    async def list_models():
        return SimpleNamespace(data=[SimpleNamespace(id=m) for m in models])

    generator._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(list=list_models),
    )
    return captured


async def _collect(generator, chunks=None, history=()):
    return [
        event
        async for event in generator.stream(
            "How long for a refund?", chunks or [_chunk()], history
        )
    ]


@pytest.mark.asyncio
async def test_adapter_streams_tokens_then_done():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    _install_fake_client(
        generator,
        chunks=[
            _delta_chunk("Within 30 "),
            _delta_chunk("days [1]."),
            _delta_chunk(finish_reason="stop"),
        ],
    )
    events = await _collect(generator)

    assert events[0] == {"type": "status", "stage": "writing"}
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "Within 30 days [1]."
    )
    assert events[-1]["type"] == "done"
    assert events[-1]["stop_reason"] == "end_turn"
    assert events[-1]["invalid_citations"] == []


@pytest.mark.asyncio
async def test_adapter_reports_hallucinated_citations_in_the_done_event():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    _install_fake_client(
        generator, chunks=[_delta_chunk("Refunds take 30 days [4]."), _delta_chunk(finish_reason="stop")]
    )
    events = await _collect(generator)
    assert events[-1]["invalid_citations"] == [4]


@pytest.mark.asyncio
async def test_adapter_maps_a_truncated_answer_to_max_tokens():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    _install_fake_client(
        generator, chunks=[_delta_chunk("Within 30"), _delta_chunk(finish_reason="length")]
    )
    events = await _collect(generator)
    assert events[-1]["stop_reason"] == "max_tokens"


@pytest.mark.asyncio
async def test_adapter_auto_detects_the_loaded_local_model():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    captured = _install_fake_client(
        generator,
        chunks=[_delta_chunk("hi"), _delta_chunk(finish_reason="stop")],
        models=("gemma-4-E4B-it-MLX-4bit",),
    )
    await _collect(generator)
    assert captured["model"] == "gemma-4-E4B-it-MLX-4bit"


@pytest.mark.asyncio
async def test_local_providers_do_not_request_usage_in_stream():
    """LM Studio and Ollama reject stream_options; hosted APIs expect it."""
    local = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    captured = _install_fake_client(local, chunks=[_delta_chunk(finish_reason="stop")])
    await _collect(local)
    assert "stream_options" not in captured

    hosted = OpenAICompatGenerator(Settings(llm_provider="groq", llm_api_key="k"))
    captured = _install_fake_client(hosted, chunks=[_delta_chunk(finish_reason="stop")])
    await _collect(hosted)
    assert captured["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_adapter_records_usage_when_the_provider_sends_it():
    generator = OpenAICompatGenerator(Settings(llm_provider="groq", llm_api_key="k"))
    _install_fake_client(
        generator,
        chunks=[
            _delta_chunk("hi"),
            _delta_chunk(finish_reason="stop"),
            _delta_chunk(usage=SimpleNamespace(prompt_tokens=120, completion_tokens=8)),
        ],
    )
    events = await _collect(generator)
    assert events[-1]["usage"]["input_tokens"] == 120
    assert events[-1]["usage"]["output_tokens"] == 8


@pytest.mark.asyncio
async def test_unreachable_local_runtime_gives_actionable_advice():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    _install_fake_client(
        generator,
        error=openai.APIConnectionError(request=SimpleNamespace()),
    )
    events = await _collect(generator)
    assert events[-1]["type"] == "error"
    message = events[-1]["message"]
    assert "localhost:1234" in message
    assert "Start Server" in message or "ollama serve" in message


@pytest.mark.asyncio
async def test_a_mid_stream_failure_keeps_the_partial_answer():
    """Discarding text the user can already read would be worse than
    admitting the stream broke."""

    class _BrokenStream:
        def __aiter__(self):
            async def gen():
                yield _delta_chunk("Within 30 ")
                raise RuntimeError("connection reset")

            return gen()

    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))

    async def create(**kwargs):
        return _BrokenStream()

    async def list_models():
        return SimpleNamespace(data=[SimpleNamespace(id="local")])

    generator._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        models=SimpleNamespace(list=list_models),
    )

    events = await _collect(generator)
    assert any(e["type"] == "token" for e in events)
    assert any(e["type"] == "error" and "interrupted" in e["message"] for e in events)
    # The stream still terminates properly so the client doesn't hang.
    assert events[-1]["type"] == "done"


@pytest.mark.asyncio
async def test_no_loaded_model_is_reported_clearly():
    generator = OpenAICompatGenerator(Settings(llm_provider="lmstudio"))
    _install_fake_client(generator, models=())
    events = await _collect(generator)
    assert events[0]["type"] == "error"
    assert "no model loaded" in events[0]["message"].lower()
