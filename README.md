# 🧠 AI Learning OS

Personal learning platform with an AI mentor. Upload your notes, docs and PDFs — get a smart assistant that answers questions with citations, builds personalized learning plans, and tests your knowledge.

## Quick Start

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY or OPENAI_API_KEY

# 2. Start everything
docker compose up -d

# 3. Open the app
open http://localhost:3000
```

That's it. The backend runs migrations automatically on startup.

## What you can do

| Feature | How |
|---|---|
| **Upload knowledge** | Drop `.md` or `.pdf` files on the Knowledge page |
| **Ask AI mentor** | Chat → New Chat → ask anything from your docs |
| **Generate learning plan** | Plans → New Plan → describe your goal |
| **Take a quiz** | Quizzes → Generate Quiz → choose a document |
| **Track progress** | Progress → see stats across all activities |

## Stack

- **Backend**: FastAPI + LangGraph + LangChain (provider-agnostic)
- **Vector DB**: Qdrant — hybrid retrieval: dense + sparse (BM25/IDF) fused with RRF, then a cross-encoder reranker (stage 2)
- **Relational DB**: PostgreSQL with Alembic migrations
- **Embeddings**: `intfloat/multilingual-e5-base` (768d, multilingual, `query:`/`passage:` prefixes — runs locally, free)
- **Reranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingual cross-encoder)
- **Chunking**: heading-aware, token-accurate; recursive boundary-aware splitter (default). See [retrieval experiments](#retrieval--eval-learning-track).
- **Frontend**: Next.js 15 + Tailwind CSS
- **Infrastructure**: Docker Compose

## LLM Providers

Switch provider via `LLM_PROVIDER` env var:

```env
# Groq (default)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# Anthropic
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6

# OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Services

| Service | URL | Description |
|---|---|---|
| Frontend | http://localhost:3000 | Next.js web app |
| Backend API | http://localhost:8000 | FastAPI + auto docs |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Qdrant Dashboard | http://localhost:6333/dashboard | Vector DB UI |

## Development

For hot-reload during development, run backend and frontend locally:

```bash
# Backend
cd backend
uv sync
uv run uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

Keep PostgreSQL and Qdrant running via Docker:

```bash
docker compose up -d postgres qdrant
```

## Retrieval & eval (learning track)

Retrieval changes are driven by numbers, not vibes. A golden-set eval harness
(`backend/app/eval/`) scores every change with **Hit@k / Recall@k / MRR / nDCG@k**
on real documents, so each experiment lands with a measured before/after.

```bash
# Validate the golden set resolves to the right chunks, then run the eval
docker compose exec -T backend sh -c 'cd /app && PYTHONPATH=/app uv run python -m app.eval.resolve'
docker compose exec -T backend sh -c 'cd /app && PYTHONPATH=/app uv run python -m app.eval.run_eval --save baseline'
```

Experiments run so far (each measured, see [LEARNING_NOTES.md](LEARNING_NOTES.md)):

| # | Experiment | Verdict |
|---|---|---|
| 1 | Multilingual embedder (e5) + reranker | ✅ adopted — big dense-recall lift on the Ukrainian corpus |
| 2 | HyDE (hypothetical doc embeddings) | ❌ off — no gain at ceiling, adds drift |
| 3 | Recursive boundary-aware chunking | ✅ on — Δ≈0 on retrieval but cleaner LLM context, free |
| 4 | Semantic chunking | ❌ off — over-fragments this corpus, strictly worse |

- **What's measured & why**: [backend/app/eval/README.md](backend/app/eval/README.md)
- **Roadmap (Tracks A→C→B)**: [NEXT_STEPS.md](NEXT_STEPS.md)

> Tunable knobs live in [backend/app/core/config.py](backend/app/core/config.py):
> `chunking_strategy` (`recursive` | `semantic` | `token`), `rerank_enabled`,
> `hyde_enabled`, `chunk_size`, `rerank_candidates`. Changing the embedder or
> chunking requires a re-ingest: `python -m app.eval.reingest`.

## Architecture

See [PLAN.md](PLAN.md) for complete architecture documentation including:
- Database schema
- LangGraph agent topologies
- API routes reference
- Document processing pipeline
