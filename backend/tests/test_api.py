"""End-to-end API tests.

The chat endpoint is exercised with a stubbed generator so the suite runs
offline and costs nothing — what we're testing is the SSE contract, not
Claude's prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main, prompts  # noqa: E402
from app.providers import anthropic_provider  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.db import connect  # noqa: E402
from app.retrieval import VectorCache  # noqa: E402


@pytest.fixture()
def client(tmp_path: Path):
    settings = Settings(
        db_path=tmp_path / "api.db",
        chunk_size=300,
        chunk_overlap=60,
        embeddings_enabled=False,
    )
    main.app.dependency_overrides[get_settings] = lambda: settings

    connection = connect(settings.db_path)
    main.app.state.conn = connection
    main.app.state.cache = VectorCache()

    # TestClient triggers lifespan, which would replace state with the real
    # db; overriding the dependency after entry keeps the temp db in play.
    with TestClient(main.app) as test_client:
        main.app.state.conn = connection
        main.app.state.cache = VectorCache()
        yield test_client

    connection.close()
    main.app.dependency_overrides.clear()


def _upload(client, name: str, body: str):
    return client.post(
        "/api/documents", files={"file": (name, body.encode("utf-8"), "text/plain")}
    )


def test_health_reports_empty_library(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents"] == 0
    assert payload["embeddings"] == "bm25-only"
    assert any("No documents" in note for note in payload["notes"])


def test_upload_then_list_then_delete(client):
    response = _upload(
        client, "policy.md", "Refunds are available within 30 days of purchase."
    )
    assert response.status_code == 201
    doc_id = response.json()["document_id"]
    assert response.json()["n_chunks"] >= 1

    listing = client.get("/api/documents").json()
    assert listing["documents"][0]["id"] == doc_id
    assert listing["total_chunks"] >= 1

    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get("/api/documents").json()["documents"] == []


def test_deleting_a_missing_document_is_404(client):
    assert client.delete("/api/documents/does-not-exist").status_code == 404


def test_reuploading_identical_content_is_a_no_op(client):
    body = "Express shipping arrives the next business day."
    first = _upload(client, "ship.md", body).json()
    second = _upload(client, "ship.md", body).json()
    assert second["skipped"] is True
    assert second["document_id"] == first["document_id"]
    assert client.get("/api/documents").json()["total_chunks"] == first["n_chunks"]


def test_reuploading_changed_content_replaces_the_old_chunks(client):
    _upload(client, "ship.md", "Express shipping arrives the next business day.")
    updated = _upload(client, "ship.md", "Express shipping now takes two days.").json()
    assert updated["replaced"] is True

    listing = client.get("/api/documents").json()
    assert len(listing["documents"]) == 1
    # The superseded text must no longer be retrievable.
    hits = client.get("/api/search", params={"q": "next business day"}).json()
    assert all("next business day" not in r["text"] for r in hits["results"])


def test_unsupported_file_type_is_415(client):
    response = client.post(
        "/api/documents", files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")}
    )
    assert response.status_code == 415


def test_empty_upload_is_400(client):
    response = client.post(
        "/api/documents", files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 400


def test_search_returns_ranked_cards(client):
    _upload(client, "refunds.md", "Refunds are issued within 30 days. Code ERR_4417.")
    _upload(client, "shipping.md", "Standard shipping takes five business days.")

    results = client.get("/api/search", params={"q": "ERR_4417"}).json()["results"]
    assert results
    assert results[0]["id"] == 1
    assert "ERR_4417" in results[0]["text"]
    assert results[0]["matched_by"] == ["keyword"]


def test_search_with_empty_query_is_400(client):
    assert client.get("/api/search", params={"q": "  "}).status_code == 400


def test_chat_streams_sources_then_tokens_then_done(client, monkeypatch):
    _upload(client, "refunds.md", "Refunds are issued within 30 days of purchase.")

    class StubGenerator:
        async def stream(self, question, chunks, history):
            assert chunks, "retrieval should have supplied context"
            yield {"type": "status", "stage": "writing"}
            yield {"type": "token", "text": "Within 30 days "}
            yield {"type": "token", "text": "[1]."}
            yield {
                "type": "done",
                "model": "stub",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2,
                          "cache_read_input_tokens": 0},
            }

    monkeypatch.setattr(main, "get_generator", lambda settings: StubGenerator())

    with client.stream(
        "POST", "/api/chat", json={"question": "What is the refund window?"}
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[0]["type"] == "sources"
    assert events[0]["sources"][0]["id"] == 1
    assert [e["type"] for e in events].count("token") == 2
    assert "".join(e["text"] for e in events if e["type"] == "token") == (
        "Within 30 days [1]."
    )
    assert events[-1]["type"] == "done"


def test_chat_still_streams_when_nothing_is_retrieved(client, monkeypatch):
    class StubGenerator:
        async def stream(self, question, chunks, history):
            assert chunks == []
            yield {"type": "token", "text": "I don't have anything on that."}
            yield {"type": "done", "model": "stub", "stop_reason": "end_turn",
                   "usage": {"input_tokens": 1, "output_tokens": 1,
                             "cache_read_input_tokens": 0}}

    monkeypatch.setattr(main, "get_generator", lambda settings: StubGenerator())

    with client.stream("POST", "/api/chat", json={"question": "anything"}) as response:
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert events[0] == {"type": "sources", "sources": []}
    assert events[-1]["type"] == "done"


def test_chat_surfaces_generator_errors_as_stream_events(client, monkeypatch):
    _upload(client, "refunds.md", "Refunds are issued within 30 days.")

    class ExplodingGenerator:
        async def stream(self, question, chunks, history):
            yield {"type": "status", "stage": "writing"}
            raise RuntimeError("upstream exploded")

    monkeypatch.setattr(main, "get_generator", lambda s: ExplodingGenerator())

    with client.stream("POST", "/api/chat", json={"question": "refunds?"}) as response:
        events = [
            json.loads(line[len("data: "):])
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]
    assert events[-1]["type"] == "error"
    assert "upstream exploded" in events[-1]["message"]


def test_chat_rejects_an_empty_question(client):
    assert client.post("/api/chat", json={"question": "   "}).status_code == 400


def test_missing_credentials_produce_a_readable_message():
    """The SDK's own wording is unhelpful; make sure we translate it."""
    exc = TypeError(
        "Could not resolve authentication method. Expected one of api_key, "
        "auth_token, or credentials to be set."
    )
    assert anthropic_provider._friendly_error(exc) == anthropic_provider.MISSING_CREDENTIALS
    # ...and that this is not mistaken for a beta-parameter problem, which
    # would send it down the retry path instead of surfacing to the user.
    assert anthropic_provider._is_fallback_rejection(exc) is False


def test_fallback_rejection_is_detected_for_retry():
    import anthropic

    assert anthropic_provider._is_fallback_rejection(
        TypeError("stream() got an unexpected keyword argument 'fallbacks'")
    )
    assert not anthropic_provider._is_fallback_rejection(RuntimeError("fallbacks"))
    assert not anthropic_provider._is_fallback_rejection(TypeError("something unrelated"))


def test_prompt_puts_sources_in_the_user_turn_not_the_system_prompt():
    """The system prompt must stay byte-identical across requests so it
    remains a valid prompt-cache prefix."""
    from app.retrieval import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id=1, document_id="d", document_title="Doc", source="d.md",
        ordinal=0, text="Refunds within 30 days.", score=1.0,
    )
    messages = prompts.build_messages("How long?", [chunk], [])
    assert messages[-1]["role"] == "user"
    assert "Refunds within 30 days." in messages[-1]["content"]
    assert 'source id="1"' in messages[-1]["content"]
    assert "Refunds within 30 days." not in prompts.SYSTEM_PROMPT


def test_prompt_keeps_history_but_does_not_reattach_old_sources():
    from app.retrieval import RetrievedChunk

    chunk = RetrievedChunk(
        chunk_id=2, document_id="d", document_title="Doc", source="d.md",
        ordinal=0, text="Express is next-day.", score=1.0,
    )
    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
    ]
    messages = prompts.build_messages("And shipping?", [chunk], history)
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "Earlier question"
    assert "<source" not in messages[0]["content"]
