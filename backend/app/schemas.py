"""Request/response models. FastAPI derives the OpenAPI schema from these."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    history: List[Turn] = Field(default_factory=list, max_length=20)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceCard(BaseModel):
    """One retrieved chunk, as the UI renders it."""

    id: int = Field(..., description="Citation number the model uses, 1-based")
    chunk_id: int
    document_id: str
    document_title: str
    source: str
    ordinal: int
    text: str
    score: float
    matched_by: List[str] = Field(
        default_factory=list,
        description="Which retrieval strategies surfaced this chunk",
    )


class SearchResponse(BaseModel):
    question: str
    results: List[SourceCard]


class DocumentSummary(BaseModel):
    id: str
    title: str
    source: str
    n_chunks: int
    n_chars: int
    created_at: str


class DocumentList(BaseModel):
    documents: List[DocumentSummary]
    total_chunks: int


class IngestResponse(BaseModel):
    document_id: str
    title: str
    n_chunks: int
    n_chars: int
    embedded: bool
    replaced: bool
    skipped: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    provider: str = Field(..., description="Active LLM provider, e.g. lmstudio")
    model: str
    local_generation: bool = Field(
        ..., description="True when generation runs on this machine, no API calls"
    )
    documents: int
    chunks: int
    embeddings: Literal["ready", "bm25-only"]
    embedding_model: Optional[str] = None
    notes: List[str] = Field(default_factory=list)
