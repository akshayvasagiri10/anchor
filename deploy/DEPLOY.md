# Deploying Anchor for free

The short version: **Vercel hosts the frontend, Hugging Face Spaces hosts the
backend, and generation goes to a free hosted API.** Total cost ₹0.

## Why not put everything on Vercel

Worth knowing rather than discovering at 2am:

| Constraint | Vercel Functions | What Anchor needs |
|---|---|---|
| Bundle size | ~250 MB | 231 MB of deps + app + model weights |
| Filesystem | Ephemeral, `/tmp` only | A SQLite index that survives requests |
| GPU | None | None for embeddings, but see below |
| Timeout | Seconds | Fine for hosted LLMs, not for local ones |

The filesystem is the real blocker, not the size. A serverless function gets a
fresh container, so your index would vanish between invocations — every
question would query an empty database.

**And a local model on Vercel is not possible at any tier.** A 3B model at Q4
is ~2 GB of weights, roughly 8× the entire function budget, and CPU inference
on a serverless vCPU would exceed the timeout mid-answer. If you want local
weights, they run on hardware you control: your Mac, or a VPS.

## Step 1 — backend on Hugging Face Spaces

Free tier: 2 vCPU, 16 GB RAM, Docker, no cold-start billing.

1. Create a Space at huggingface.co/new-space → **Docker** → **Blank**.
2. Push this repo to it. The Space needs `Dockerfile` at its root, so either
   move `backend/Dockerfile` up a level, or point the Space at a subdirectory
   in its settings.
3. Copy `deploy/SPACE_README.md` to the Space's `README.md` — the YAML
   frontmatter is what tells HF the SDK is Docker and the port is 7860.
4. Under **Settings → Variables and secrets**, add:

   | Name | Value |
   |---|---|
   | `ANCHOR_LLM_PROVIDER` | `groq` |
   | `GROQ_API_KEY` | your free key from console.groq.com |
   | `ANCHOR_CORS_ORIGINS` | your Vercel URL (fill in after step 2) |

Your API is then at `https://<user>-<space>.hf.space`. Check
`https://<user>-<space>.hf.space/api/health`.

> `lmstudio` and `ollama` are invalid here — they resolve to localhost, which
> inside a container is the container.

**Free Spaces have ephemeral disk.** The Dockerfile bakes the sample corpus in
and indexes it at build time, so a restarted Space always demos correctly, but
documents uploaded through the UI are lost on restart. That's a fine trade for
a portfolio demo; attach persistent storage if it isn't.

## Step 2 — frontend on Vercel

1. Import the repo at vercel.com/new.
2. Set **Root Directory** to `frontend`.
3. Add an environment variable:

   | Name | Value |
   |---|---|
   | `NEXT_PUBLIC_API_URL` | `https://<user>-<space>.hf.space` |

4. Deploy, then go back and put the resulting Vercel URL into the Space's
   `ANCHOR_CORS_ORIGINS`. Forgetting this is the #1 cause of a deployed
   frontend that loads but shows "Backend unreachable" — the browser blocks
   the request, and nothing appears in the backend logs.

## Step 3 — check it

```bash
curl https://<user>-<space>.hf.space/api/health
curl "https://<user>-<space>.hf.space/api/search?q=refund"
```

`/api/search` needs no LLM and no key, so it isolates retrieval from
generation. If search works and chat doesn't, the problem is your provider
credentials, not your deployment.

## Free generation providers

| Provider | Free tier | Notes |
|---|---|---|
| **Groq** | Generous, very fast | Best default. Llama 3.3 70B at high tokens/sec |
| **OpenRouter** | Models with a `:free` suffix | Wide model choice, slower, stricter limits |
| **HF Inference** | Small hourly quota | Convenient if you're already on HF |

Set `ANCHOR_LLM_PROVIDER` and the matching key; nothing else changes.

## Running the container locally first

```bash
cd anchor
docker build -f backend/Dockerfile -t anchor-api .
docker run --rm -p 7860:7860 \
  -e ANCHOR_LLM_PROVIDER=groq -e GROQ_API_KEY=gsk_... \
  anchor-api
```

Then hit `http://localhost:7860/api/health`. If it works here it works on
Spaces — the environment is the same.
