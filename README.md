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
- **Vector DB**: Qdrant (dense + sparse hybrid search with RRF)
- **Relational DB**: PostgreSQL with Alembic migrations
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (runs locally, free)
- **Frontend**: Next.js 15 + Tailwind CSS
- **Infrastructure**: Docker Compose

## LLM Providers

Switch provider via `LLM_PROVIDER` env var:

```env
# Anthropic (default)
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

## Architecture

See [PLAN.md](PLAN.md) for complete architecture documentation including:
- Database schema
- LangGraph agent topologies
- API routes reference
- Document processing pipeline
