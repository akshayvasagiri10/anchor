# ⚓ Anchor

**A RAG chatbot that answers only from your documents — and proves it.**
**Runs fully offline on your laptop. No API key required.**

Most "chat with your PDF" projects embed some chunks, run a cosine search, and
hand the top 5 to an LLM. That works until someone asks about an error code, a
part number, or a proper noun the embedding model has never seen — at which
point dense-only retrieval quietly returns the wrong passage and the model
answers confidently from it.

Anchor uses **hybrid retrieval**: BM25 for exact terms, dense vectors for
meaning, fused with Reciprocal Rank Fusion. Every answer carries inline
citations that resolve to the exact chunk they came from.

```
┌──────────────┐   ┌──────────────────────────────────────┐   ┌──────────────────┐
│  Next.js 15  │──▶│  FastAPI                             │──▶│ LM Studio        │
│  streaming   │   │                                      │   │ Ollama           │
│  chat + SSE  │◀──│  ┌────────────┐   ┌───────────────┐  │◀──│ Groq / OpenRouter│
└──────────────┘   │  │ BM25       │   │ dense cosine  │  │   │ Claude           │
                   │  │ SQLite FTS5│   │ MiniLM (ONNX) │  │   └──────────────────┘
                   │  └─────┬──────┘   └───────┬───────┘  │    one adapter, all
                   │        └────── RRF ───────┘          │    of them
                   │              SQLite                  │
                   └──────────────────────────────────────┘
```

Generation is pluggable. Anchor ships pointing at **LM Studio on localhost**,
so a fresh clone works with no key and no network. LM Studio, Ollama, Groq,
OpenRouter, vLLM and OpenAI all speak the same wire format, so all of them are
one adapter and switching is a `.env` line:

```bash
ANCHOR_LLM_PROVIDER=lmstudio    # local, free, offline  (default)
ANCHOR_LLM_PROVIDER=ollama      # local, free, offline
ANCHOR_LLM_PROVIDER=groq        # hosted, free tier, very fast
ANCHOR_LLM_PROVIDER=anthropic   # hosted, best grounding discipline
```

---

## Why hybrid retrieval

The two strategies fail in **opposite** directions, which is exactly why
fusing them works. Measured against the sample corpus in `data/samples/`:

| Query | BM25 alone | Dense alone | Hybrid (RRF) |
|---|---|---|---|
| `ERR_4417` | ✅ rank 1 | ⚠️ rank 2 | ✅ rank 1 |
| "how long do I have to get my money back" | ❌ no term overlap with *refund* | ✅ rank 1 | ✅ rank 1 |
| "how do I roll back a bad deploy" | ✅ rank 1 | ✅ rank 1 | ✅ rank 1 |

BM25 nails rare literals — error codes, SKUs, function names — and whiffs on
paraphrase. Dense vectors handle paraphrase and whiff on literals they never
saw in pretraining. RRF recovers both.

RRF also sidesteps the usual problem with combining them. BM25 produces an
unbounded negative log-scale score; cosine produces `[-1, 1]`. Normalising
those onto a common scale requires a tuning constant that drifts with your
corpus. RRF ranks instead of scores:

```
score(d) = Σ  1 / (k + rank_s(d))       for each strategy s,  k = 60
```

No normalisation, no mixing weight, no per-corpus tuning.

## What else is in here

- **Section-aware chunking.** Markdown headings are hard boundaries, and every
  chunk carries its heading breadcrumb (`Handbook › Refunds › Digital goods`).
  That prefix does real work: BM25 can now match "refund" against a chunk
  whose body never repeats the word, and the embedding gains topical context
  it would otherwise have to infer. On the sample corpus this took retrieval
  from *"both giant chunks always match"* to the correct section every time.
- **No vector database.** SQLite holds the text, an FTS5 external-content
  index gives Okapi BM25 via the built-in `bm25()` function, and embeddings
  live in a `BLOB` column as normalised float32. 50k chunks × 384 dims is a
  73 MB matrix and one matmul per query. An ANN index would add a dependency
  and an accuracy loss to solve a problem this project does not have.
- **No embeddings API.** `all-MiniLM-L6-v2` runs locally on onnxruntime via
  `fastembed` — no second API key, no per-query cost, works offline. Choosing
  ONNX over sentence-transformers dropped the install from **689 MB to 231 MB**
  by cutting torch entirely, which is the difference between deployable and
  not. Same 384-dim vectors either way. If it isn't installed, retrieval
  degrades to BM25-only rather than crashing, and `/api/health` says so.
- **Hallucinated citations get named.** Every answer is scanned for citation
  numbers that don't exist, and the UI flags them. This is not decoration:
  point Anchor at a 4B local model and you will watch it cite `[9]` when it
  was handed four sources. A citation nobody checks is just a confident
  guess — so Anchor checks.
- **Grounding you can audit.** Sources stream to the client *before* the first
  token, so `[3]` is already clickable when the model writes it. A citation
  outside the source range renders as plain text, not a dead link — hiding an
  invented citation would hide a real failure.
- **Idempotent ingestion.** Content-hashed, so re-uploading an unchanged file
  is a no-op and re-uploading a changed one supersedes the old chunks instead
  of leaving stale text to be retrieved alongside the new.

---

## Quickstart

Requires Python 3.9+ and Node 18+.

```bash
git clone <your-repo-url> anchor && cd anchor
cp .env.example .env
```

**Pick a model.** The default is LM Studio — install it, download a model, and
hit *Start Server* in the Developer tab. Anchor auto-detects whichever model
you loaded. Ollama works identically:

```bash
brew install ollama && ollama serve
ollama pull qwen2.5:7b-instruct
# then set ANCHOR_LLM_PROVIDER=ollama in .env
```

> **Pick 7B or larger if you can.** Anchor's whole discipline is *refuse when
> the sources don't cover it, cite accurately* — and that is precisely where
> small models fall down. A 3–4B model will invent citations and answer from
> pretraining rather than admitting a gap. Anchor detects and flags this
> rather than hiding it, but detection is not a fix. On 16 GB of unified
> memory, Qwen2.5 7B at Q4 is comfortable.

**Backend**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/ingest_cli.py ../data/samples   # index the sample corpus
uvicorn app.main:app --reload                  # http://localhost:8000
```

**Frontend**

```bash
cd frontend
npm install
npm run dev                                    # http://localhost:3000
```

Then ask *"What does error ERR_4417 mean?"* or *"Can I ship to a PO box?"*

> The first ingest downloads the ~90 MB embedding model. Subsequent runs are
> instant. To skip it entirely, pass `--no-embed` or set
> `ANCHOR_EMBEDDINGS_ENABLED=false` — you get BM25-only retrieval.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Readiness, chunk counts, retrieval mode, configuration warnings |
| `GET` | `/api/documents` | List the indexed library |
| `POST` | `/api/documents` | Upload and index a file (`.md`, `.txt`, `.pdf`, `.csv`, `.json`, `.rst`) |
| `DELETE` | `/api/documents/{id}` | Remove a document and its chunks |
| `GET` | `/api/search?q=` | **Retrieval without generation** — the endpoint you use to debug relevance |
| `POST` | `/api/chat` | Retrieve, then stream a grounded answer as SSE |

Interactive docs at `http://localhost:8000/docs`.

`/api/chat` emits these SSE event types in order:

```jsonc
{"type": "sources", "sources": [...]}          // before generation starts
{"type": "status",  "stage": "thinking"}       // thinking text is omitted by
{"type": "status",  "stage": "writing"}        //   default — this drives the spinner
{"type": "token",   "text": "Within 30 "}
{"type": "done",    "usage": {...}}
{"type": "error",   "message": "..."}          // terminal, replaces `done`
```

---

## Configuration

Everything is env-driven with `ANCHOR_`-prefixed variables (see
`.env.example`). The ones worth touching:

| Variable | Default | Notes |
|---|---|---|
| `ANCHOR_LLM_PROVIDER` | `lmstudio` | `lmstudio`,`ollama`,`groq`,`openrouter`,`openai`,`anthropic` |
| `ANCHOR_LLM_MODEL` | per provider | Blank auto-detects the loaded local model |
| `ANCHOR_LLM_BASE_URL` | per provider | Override to reach a runtime on another host |
| `GROQ_API_KEY` etc. | — | Only for hosted providers. Local ones need nothing |
| `ANCHOR_TEMPERATURE` | `0.2` | Low but non-zero — 0 makes some local runtimes loop |
| `ANCHOR_EFFORT` | `low` | Anthropic only. Latency is tuned with effort, never by disabling thinking |
| `ANCHOR_CHUNK_SIZE` / `_OVERLAP` | `1200` / `200` | Characters. ~250–350 tokens per chunk |
| `ANCHOR_RETRIEVAL_TOP_K` | `6` | Chunks passed to the model |
| `ANCHOR_CANDIDATE_POOL` | `30` | Per-strategy candidates before fusion |
| `ANCHOR_EMBEDDINGS_ENABLED` | `true` | `false` → BM25-only |

---

## Design notes

**The system prompt is a boundary, not a hint.** Most RAG hallucinations come
from a prompt that presents context as helpful background. Anchor's system
prompt states that outside knowledge is forbidden, that "the sources don't
cover this" is a *correct* answer, and that conflicting sources must be
surfaced rather than silently reconciled.

**Sources go in the user turn, not the system prompt.** The system prompt is
byte-identical on every request, so it stays a valid prompt-cache prefix while
the volatile per-query context sits after it. Putting sources in the system
prompt would invalidate the cache on every single query.

**History carries no sources.** Only the current turn gets context. Re-attaching
excerpts from earlier turns would balloon the prompt and let stale chunks
outvote the ones actually retrieved for the question being asked.

**Chunk size invariant.** Every chunk is at most `chunk_size + overlap`
characters. The slack is real and documented: a chunk is *carried overlap*
plus *fresh content*, and the packer checks its budget before adding the next
sentence, so one sentence can push past `chunk_size` before the flush. The
test suite asserts this across five pathological input shapes — including a
single token wider than a whole chunk, which is the case that broke it first.

---

## Tests

```bash
cd backend && python -m pytest tests -q     # 76 tests
```

They run offline and cost nothing — the chat endpoint is exercised with a
stubbed generator, because what's worth testing is the SSE contract and the
grounding discipline, not any model's prose. The suite covers the chunker's
size invariant, heading hierarchy, FTS5 query sanitisation (including operator
injection), RRF ordering, the BM25-only degradation path, ingest idempotency,
provider resolution across all six presets, the OpenAI-compatible adapter's
streaming contract, hallucinated-citation detection, and every API error
branch.

The adapter is additionally verified against a real HTTP server speaking the
OpenAI streaming protocol — unit stubs cannot catch a malformed request body
or an SSE parsing bug.

---

## Deploying it for free

Frontend on Vercel, backend on Hugging Face Spaces, generation on Groq's free
tier. Full walkthrough in [`deploy/DEPLOY.md`](deploy/DEPLOY.md), including
why the backend *cannot* go on Vercel (ephemeral filesystem, not size) and why
a local model cannot be deployed serverlessly at any tier.

---

## Limitations

- **Brute-force vector search.** Fine to roughly 100k chunks on a laptop.
  Beyond that, `VectorCache` is the single class to swap for an ANN index.
- **Scanned PDFs need OCR first.** `pypdf` extracts text, not pixels; the API
  returns a 422 that says so rather than indexing an empty document.
- **Single-process SQLite.** WAL mode handles concurrent reads well, but
  ingest and query share one connection. Postgres + `pgvector` is the move
  if you need real concurrency.
- **No reranker.** A cross-encoder over the fused candidates would likely be
  the single biggest quality win from here.
- **Small models are the weak link, not the retrieval.** Below ~7B, grounding
  discipline degrades sharply: invented citations, answers from pretraining,
  ignored refusal instructions. Anchor surfaces this rather than hiding it,
  but you should read the warnings when they appear.
- **No eval harness.** Retrieval quality is currently argued from examples,
  not measured. A golden Q→chunk set with recall@k in CI is the honest next
  step, and the reason it's project #2 on my list.

## License

MIT
