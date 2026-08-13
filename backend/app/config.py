"""Runtime configuration, read once from the environment.

Python 3.9 note: `from __future__ import annotations` lets us write modern
annotations (`list[str]`, `str | None`) without a 3.10+ interpreter.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, NamedTuple, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

ENV_FILES = (PROJECT_ROOT / ".env", BACKEND_ROOT / ".env")

# pydantic-settings reads .env into the Settings object, but it does NOT put
# those values into os.environ — and the LLM SDKs resolve credentials from
# os.environ, not from us. Without this, an API key written to .env is read by
# nothing and every chat request fails with "no credentials".
# override=False so a real exported variable still wins over the file.
for _env_file in ENV_FILES:
    if _env_file.exists():
        load_dotenv(_env_file, override=False)


class Preset(NamedTuple):
    """A ready-made provider configuration.

    Every one of these except `anthropic` speaks the OpenAI chat-completions
    wire format, so a single adapter covers local runtimes and hosted APIs
    alike — switching between them is a .env change, not a code change.
    """

    base_url: Optional[str]
    default_model: str
    api_key_env: Optional[str]
    # Local runtimes ignore the key but the SDK still requires a non-empty
    # string, so we supply a placeholder.
    placeholder_key: Optional[str]
    # `stream_options.include_usage` is standard on hosted APIs and patchy on
    # local ones; asking for it where it isn't supported fails the request.
    usage_in_stream: bool
    local: bool


PRESETS: Dict[str, Preset] = {
    "lmstudio": Preset(
        base_url="http://localhost:1234/v1",
        default_model="",  # auto-detected from /v1/models
        api_key_env=None,
        placeholder_key="lm-studio",
        usage_in_stream=False,
        local=True,
    ),
    "ollama": Preset(
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5:7b-instruct",
        api_key_env=None,
        placeholder_key="ollama",
        usage_in_stream=False,
        local=True,
    ),
    "groq": Preset(
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        placeholder_key=None,
        usage_in_stream=True,
        local=False,
    ),
    "openrouter": Preset(
        base_url="https://openrouter.ai/api/v1",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        api_key_env="OPENROUTER_API_KEY",
        placeholder_key=None,
        usage_in_stream=True,
        local=False,
    ),
    "openai": Preset(
        base_url=None,  # the SDK's own default
        default_model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        placeholder_key=None,
        usage_in_stream=True,
        local=False,
    ),
    "anthropic": Preset(
        base_url=None,
        default_model="claude-opus-5",
        api_key_env="ANTHROPIC_API_KEY",
        placeholder_key=None,
        usage_in_stream=True,
        local=False,
    ),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILES,
        env_prefix="ANCHOR_",
        extra="ignore",
    )

    # --- Storage -----------------------------------------------------------
    db_path: Path = PROJECT_ROOT / "data" / "anchor.db"

    # --- Chunking ----------------------------------------------------------
    # Characters, not tokens: a chunker that respects sentence boundaries at
    # ~1200 chars lands around 250-350 tokens, which is the sweet spot for
    # retrieval granularity without shredding context.
    chunk_size: int = 1200
    chunk_overlap: int = 200
    min_chunk_size: int = 120

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 6
    candidate_pool: int = 30  # per-strategy candidates before fusion
    rrf_k: int = 60  # Reciprocal Rank Fusion damping constant

    # --- Embeddings --------------------------------------------------------
    # fastembed runs this exact model on onnxruntime — same 384-dim, already
    # L2-normalized vectors as sentence-transformers, without pulling torch.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embeddings_enabled: bool = True

    # --- Generation --------------------------------------------------------
    # Defaults to your local LM Studio, so a fresh clone answers questions
    # with no API key and no network. Set ANCHOR_LLM_PROVIDER to switch.
    llm_provider: str = "lmstudio"
    llm_model: str = ""  # blank -> the preset default, or auto-detect locally
    llm_base_url: str = ""  # blank -> the preset default
    llm_api_key: str = ""  # blank -> read from the preset's env var
    max_tokens: int = 2048
    temperature: float = 0.2

    # Anthropic-only knobs, ignored by the OpenAI-compatible providers.
    effort: str = "low"

    # --- Server ------------------------------------------------------------
    cors_origins: str = "http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def preset(self) -> Preset:
        try:
            return PRESETS[self.llm_provider]
        except KeyError:
            raise ValueError(
                f"Unknown ANCHOR_LLM_PROVIDER '{self.llm_provider}'. "
                f"Choose one of: {', '.join(sorted(PRESETS))}"
            ) from None

    @property
    def resolved_model(self) -> str:
        return self.llm_model or self.preset.default_model

    @property
    def resolved_base_url(self) -> Optional[str]:
        return self.llm_base_url or self.preset.base_url

    @property
    def resolved_api_key(self) -> Optional[str]:
        """Explicit setting, then the provider's conventional env var, then a
        placeholder for local runtimes that don't check it."""
        if self.llm_api_key:
            return self.llm_api_key
        preset = self.preset
        if preset.api_key_env:
            key = os.environ.get(preset.api_key_env)
            if key:
                return key
        return preset.placeholder_key

    @property
    def credentials_present(self) -> bool:
        """Local runtimes need no credentials; hosted ones do."""
        return self.preset.local or bool(self.resolved_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
