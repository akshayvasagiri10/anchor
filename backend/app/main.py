"""FastAPI application: ingest, search, and streaming chat."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from . import __version__
from . import embeddings as emb
from . import ingest as ingest_mod
from .config import Settings, get_settings
from .db import assert_fts5, connect
from .providers import get_generator
from .retrieval import RetrievedChunk, VectorCache, search
from .schemas import (
    ChatRequest,
    DocumentList,
    DocumentSummary,
    HealthResponse,
    IngestResponse,
    SearchResponse,
    SourceCard,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
)
logger = logging.getLogger("anchor")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    conn = connect(settings.db_path)
    assert_fts5(conn)
    app.state.conn = conn
    app.state.cache = VectorCache()
    logger.info("Anchor %s ready — db=%s", __version__, settings.db_path)
    # The embedding model loads on first use, not here: a cold start that
    # blocks for 10s on a torch import is a bad developer experience, and a
    # BM25-only server is still useful while it warms.
    try:
        yield
    finally:
        conn.close()


app = FastAPI(
    title="Anchor",
    version=__version__,
    description="A hybrid-retrieval RAG chatbot that grounds every answer in sources.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Dependencies ---------------------------------------------------------


def get_conn() -> sqlite3.Connection:
    return app.state.conn


def get_cache() -> VectorCache:
    return app.state.cache


def get_embedder(settings: Settings = Depends(get_settings)) -> Optional[emb.Embedder]:
    return emb.get_embedder(
        settings.embedding_model, enabled=settings.embeddings_enabled
    )


# --- Health ---------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse)
def health(
    conn: sqlite3.Connection = Depends(get_conn),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    n_docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
    n_chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    loaded, attempted, embed_error = emb.embedding_status()
    notes: list[str] = []
    degraded = False

    if not settings.embeddings_enabled:
        mode = "bm25-only"
        notes.append("Dense embeddings are disabled; retrieval is BM25 only.")
    elif embed_error:
        mode = "bm25-only"
        degraded = True
        notes.append(embed_error)
    else:
        # Enabled and nothing has failed. Not being loaded yet is expected on
        # a cold start, not a degradation.
        mode = "ready"
        if not attempted:
            notes.append(
                "Embedding model loads on first use — the first upload or "
                "query will take a few extra seconds."
            )

    try:
        preset = settings.preset
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not settings.credentials_present:
        degraded = True
        notes.append(
            f"No credentials for '{settings.llm_provider}'. Set "
            f"{preset.api_key_env} in your environment or .env file. Search "
            f"works regardless; only chat needs this."
        )
    if preset.local:
        notes.append(
            f"Generation runs locally via {settings.llm_provider} at "
            f"{settings.resolved_base_url} — make sure it is running with a "
            f"model loaded."
        )
    if n_docs == 0:
        notes.append("No documents ingested yet. Upload one to get started.")

    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=__version__,
        provider=settings.llm_provider,
        model=settings.resolved_model or "(auto-detect)",
        local_generation=preset.local,
        documents=n_docs,
        chunks=n_chunks,
        embeddings=mode,
        embedding_model=settings.embedding_model if mode == "ready" else None,
        notes=notes,
    )


# --- Documents ------------------------------------------------------------


@app.get("/api/documents", response_model=DocumentList)
def list_documents(conn: sqlite3.Connection = Depends(get_conn)) -> DocumentList:
    rows = conn.execute(
        """
        SELECT id, title, source, n_chunks, n_chars, created_at
        FROM documents ORDER BY created_at DESC
        """
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    return DocumentList(
        documents=[DocumentSummary(**dict(r)) for r in rows], total_chunks=total
    )


@app.post("/api/documents", response_model=IngestResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    conn: sqlite3.Connection = Depends(get_conn),
    cache: VectorCache = Depends(get_cache),
    settings: Settings = Depends(get_settings),
    embedder: Optional[emb.Embedder] = Depends(get_embedder),
) -> IngestResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )

    filename = file.filename or "upload.txt"
    try:
        text = ingest_mod.extract_text(filename, data)
    except ingest_mod.UnsupportedFileType as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted. Scanned PDFs need OCR first.",
        )

    try:
        result = ingest_mod.ingest_text(
            conn,
            cache,
            settings,
            title=filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " "),
            source=filename,
            text=text,
            embedder=embedder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return IngestResponse(**vars(result))


@app.delete("/api/documents/{document_id}", status_code=204, response_class=Response)
def delete_document(
    document_id: str,
    conn: sqlite3.Connection = Depends(get_conn),
    cache: VectorCache = Depends(get_cache),
) -> Response:
    if not ingest_mod.delete_document(conn, cache, document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
    return Response(status_code=204)


# --- Retrieval ------------------------------------------------------------


def _run_search(
    conn: sqlite3.Connection,
    cache: VectorCache,
    settings: Settings,
    embedder: Optional[emb.Embedder],
    question: str,
    top_k: Optional[int],
) -> list[SourceCard]:
    chunks = search(
        conn,
        cache,
        embedder,
        question,
        top_k=top_k or settings.retrieval_top_k,
        candidate_pool=settings.candidate_pool,
        rrf_k=settings.rrf_k,
    )
    return [
        SourceCard(
            id=i,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            document_title=c.document_title,
            source=c.source,
            ordinal=c.ordinal,
            text=c.text,
            score=round(c.score, 6),
            matched_by=c.matched_by,
        )
        for i, c in enumerate(chunks, start=1)
    ]


@app.get("/api/search", response_model=SearchResponse)
def search_endpoint(
    q: str,
    top_k: Optional[int] = None,
    conn: sqlite3.Connection = Depends(get_conn),
    cache: VectorCache = Depends(get_cache),
    settings: Settings = Depends(get_settings),
    embedder: Optional[emb.Embedder] = Depends(get_embedder),
) -> SearchResponse:
    """Retrieval without generation — the endpoint you use to debug relevance."""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    return SearchResponse(
        question=q, results=_run_search(conn, cache, settings, embedder, q, top_k)
    )


# --- Chat -----------------------------------------------------------------


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    conn: sqlite3.Connection = Depends(get_conn),
    cache: VectorCache = Depends(get_cache),
    settings: Settings = Depends(get_settings),
    embedder: Optional[emb.Embedder] = Depends(get_embedder),
) -> StreamingResponse:
    """Retrieve, then stream a grounded answer as Server-Sent Events.

    Sources are sent as the first event so the UI can render citation targets
    before the first token arrives — the answer references [1] within a few
    hundred milliseconds, and the card is already on screen.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    sources = _run_search(
        conn, cache, settings, embedder, question, request.top_k
    )
    generator = get_generator(settings)
    history = [t.model_dump() for t in request.history]

    context = [_card_to_chunk(card) for card in sources]

    async def event_stream() -> AsyncIterator[str]:
        yield _sse({"type": "sources", "sources": [s.model_dump() for s in sources]})
        try:
            async for event in generator.stream(question, context, history):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - never leave the stream hanging
            logger.exception("Chat stream failed")
            yield _sse({"type": "error", "message": f"Unexpected error: {exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # stop nginx from buffering the stream
        },
    )


def _card_to_chunk(card: SourceCard) -> RetrievedChunk:
    """Adapt the API-facing card back to the retrieval dataclass the
    generator expects. Keeps the LLM layer free of HTTP concerns."""
    return RetrievedChunk(
        chunk_id=card.chunk_id,
        document_id=card.document_id,
        document_title=card.document_title,
        source=card.source,
        ordinal=card.ordinal,
        text=card.text,
        score=card.score,
    )
