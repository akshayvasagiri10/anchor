.PHONY: help setup backend frontend ingest test docker clean

help:
	@echo "make setup     — create the venv, install backend + frontend deps"
	@echo "make ingest    — index data/samples into the SQLite store"
	@echo "make backend   — run the API on :8000"
	@echo "make frontend  — run the Next.js app on :3000"
	@echo "make test      — run the backend test suite"
	@echo "make docker    — build the deployable backend image"
	@echo "make clean     — drop the index and build artefacts"

setup:
	cd backend && python3 -m venv .venv && .venv/bin/pip install -q -r requirements.txt
	cd frontend && npm install
	@test -f .env || cp .env.example .env
	@echo "\nReady. Anchor defaults to LM Studio on localhost — no API key needed."
	@echo "Start LM Studio (Developer tab -> Start Server), then: make ingest && make backend"

ingest:
	cd backend && .venv/bin/python scripts/ingest_cli.py ../data/samples

backend:
	cd backend && .venv/bin/python -m uvicorn app.main:app --reload

frontend:
	cd frontend && npm run dev

test:
	cd backend && .venv/bin/python -m pytest tests -q

docker:
	docker build -f backend/Dockerfile -t anchor-api .
	@echo "\nRun it:  docker run --rm -p 7860:7860 -e ANCHOR_LLM_PROVIDER=groq -e GROQ_API_KEY=... anchor-api"

clean:
	rm -f data/anchor.db data/anchor.db-wal data/anchor.db-shm
	rm -rf frontend/.next backend/.pytest_cache
	find backend -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
