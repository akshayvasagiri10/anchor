"""Dense embeddings, loaded lazily and degrading gracefully.

Anchor embeds locally — no second API key, no per-query cost, works offline.
The runtime is `fastembed`, which executes the same `all-MiniLM-L6-v2` model
on onnxruntime. That choice is what makes the backend deployable: the
sentence-transformers path pulled in torch at 339 MB, against roughly 50 MB
for onnxruntime, and both produce identical 384-dimension L2-normalized
vectors — so an index built with either one stays valid.

If fastembed isn't installed, `get_embedder()` returns None and retrieval
falls back to BM25 alone. That's a real, useful mode, not a broken one, so we
surface it in /api/health rather than crashing.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_embedder: Optional["Embedder"] = None
_load_attempted = False
_load_error: Optional[str] = None


class Embedder:
    """Thin wrapper that guarantees float32, L2-normalized output.

    Normalizing at write time means cosine similarity is a plain dot product
    at query time — one matmul, no per-row division.
    """

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # local import: keeps startup cold

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # Probe once rather than trusting a hard-coded dimension: a user who
        # points ANCHOR_EMBEDDING_MODEL at a different model should get the
        # right shape, not a reshape error 200 chunks into an ingest.
        probe = np.asarray(next(iter(self._model.embed(["dimension probe"]))))
        self.dim = int(probe.shape[0])

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._model.embed(list(texts), batch_size=batch_size))
        matrix = np.asarray(vectors, dtype=np.float32)
        # fastembed normalizes already; this is belt-and-braces so a swapped
        # model can never silently break the dot-product-is-cosine assumption.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        np.divide(matrix, np.where(norms == 0, 1.0, norms), out=matrix)
        return matrix

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def get_embedder(model_name: str, *, enabled: bool = True) -> Optional[Embedder]:
    """Load the embedding model once, or return None if unavailable."""
    global _embedder, _load_attempted, _load_error

    if not enabled:
        return None
    if _embedder is not None:
        return _embedder

    with _lock:
        if _embedder is not None:
            return _embedder
        if _load_attempted:
            return None
        _load_attempted = True
        try:
            logger.info("Loading embedding model %s (first use)", model_name)
            _embedder = Embedder(model_name)
            logger.info("Embedding model ready, dim=%d", _embedder.dim)
        except ImportError as exc:
            _load_error = (
                "fastembed is not installed; retrieval is running in BM25-only "
                "mode. Install it with: pip install fastembed"
            )
            logger.warning("%s (%s)", _load_error, exc)
        except Exception as exc:  # noqa: BLE001 - model download/load can fail many ways
            _load_error = f"Failed to load embedding model {model_name}: {exc}"
            logger.warning(_load_error)
    return _embedder


def embedding_status() -> tuple[bool, bool, Optional[str]]:
    """(loaded, load_attempted, error) — for the health endpoint.

    The distinction matters: "not loaded yet" is normal on a cold start and
    should not be reported as a degraded service, whereas "tried and failed"
    genuinely means retrieval is running on BM25 alone.
    """
    return _embedder is not None, _load_attempted, _load_error


def pack(vector: np.ndarray) -> bytes:
    """float32 little-endian blob, ready for a SQLite BLOB column."""
    return np.ascontiguousarray(vector, dtype="<f4").tobytes()


def unpack(blob: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").reshape(dim)


def unpack_matrix(blobs: Sequence[bytes], dim: int) -> np.ndarray:
    """Stack many blobs into an (n, dim) matrix in one allocation."""
    if not blobs:
        return np.zeros((0, dim), dtype=np.float32)
    buf = b"".join(blobs)
    return np.frombuffer(buf, dtype="<f4").reshape(len(blobs), dim)


def reset_for_tests() -> None:
    """Test hook — clears the module-level singleton."""
    global _embedder, _load_attempted, _load_error
    with _lock:
        _embedder = None
        _load_attempted = False
        _load_error = None
