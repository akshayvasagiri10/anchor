---
title: Anchor API
emoji: ⚓
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Anchor — backend API

> **Requires a Hugging Face PRO subscription.** Free accounts can only create
> Static Spaces; Docker Spaces return `402 Payment Required` on creation.
> See `deploy/DEPLOY.md` for free alternatives.

Hybrid-retrieval RAG backend: BM25 (SQLite FTS5) + dense vectors (fastembed /
all-MiniLM-L6-v2) fused with Reciprocal Rank Fusion.

Source: https://github.com/<your-username>/anchor

## Endpoints

- `GET /docs` — interactive API reference
- `GET /api/health` — readiness, chunk counts, active provider
- `GET /api/search?q=` — retrieval only, no LLM, no key required
- `POST /api/chat` — grounded answer, streamed as SSE

## Configuration

Set these as **Space secrets** (Settings → Variables and secrets):

| Secret | Value |
|---|---|
| `ANCHOR_LLM_PROVIDER` | `groq` (or `openrouter`) |
| `GROQ_API_KEY` | your free Groq key |
| `ANCHOR_CORS_ORIGINS` | your Vercel URL, e.g. `https://anchor.vercel.app` |

`lmstudio` and `ollama` are **not** valid here — they point at localhost,
which inside a container means the container itself.

## Note on persistence

Free Spaces have ephemeral disk. The sample corpus is baked into the image and
indexed at build time, so the demo always works, but documents uploaded
through the UI are lost when the Space restarts. Attach persistent storage, or
point `ANCHOR_DB_PATH` at a mounted volume, if you need uploads to survive.
