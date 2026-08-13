from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db import connect  # noqa: E402
from app.retrieval import VectorCache  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        db_path=tmp_path / "test.db",
        chunk_size=300,
        chunk_overlap=60,
        min_chunk_size=40,
        embeddings_enabled=False,
    )


@pytest.fixture()
def conn(settings: Settings):
    connection = connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture()
def cache() -> VectorCache:
    return VectorCache()
