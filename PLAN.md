# AI Learning OS — Project Plan

## Vision

Personal learning platform that transforms scattered knowledge (notes, docs, code) into a personalized learning system with an AI mentor. Unlike a basic RAG chatbot, the platform acts as a teacher — understanding knowledge gaps, building learning plans, and tracking progress.

---

## Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Python packaging | `uv` | 10–100x faster than poetry; single binary; clean Docker integration |
| DB access | SQLAlchemy 2.0 async + asyncpg | LLM calls are I/O-bound (10–30s); async prevents worker thread blocking |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Free, no API key, 384-dim, ~50ms/chunk on CPU |
| Vector search | Qdrant dense+sparse → RRF fusion | +15% recall vs pure vector search; better for specific terms |
| Migrations | Alembic | `create_all` breaks on first `ALTER TABLE`; 30 min setup saves hours later |
| Frontend | Next.js 15 App Router + Tailwind | App Router → streaming RSC; no external component deps |
| LLM abstraction | LangChain (`LLM_PROVIDER` env var) | Switch OpenAI↔Anthropic without code changes |
| Orchestration | LangGraph | Structured multi-step agents with conditional edges and retry loops |

---

## Project Structure

```
AILearningOS/
├── docker-compose.yml              # All services: postgres, qdrant, backend, frontend
├── .env.example                    # Template for environment variables
├── .gitignore
├── PLAN.md                         # This file
├── README.md                       # Getting-started guide
│
├── backend/
│   ├── Dockerfile
│   ├── pyproject.toml              # uv project config
│   ├── alembic.ini
│   ├── alembic/
│   │   ├── env.py                  # Async alembic with SQLAlchemy
│   │   ├── script.py.mako
│   │   └── versions/
│   │       └── 0001_initial.py     # Full schema migration
│   └── app/
│       ├── main.py                 # FastAPI app factory + CORS + router mounting
│       ├── core/
│       │   ├── config.py           # Pydantic Settings (env vars)
│       │   ├── database.py         # AsyncEngine, get_db() dependency
│       │   └── lifespan.py         # Startup: Qdrant collection init, embedder load, graph compile
│       ├── models/                 # SQLAlchemy ORM models
│       │   ├── document.py         # Document, DocumentChunk
│       │   ├── chat.py             # ChatSession, ChatMessage
│       │   ├── plan.py             # LearningPlan, PlanTopic (self-referential tree)
│       │   ├── quiz.py             # Quiz, QuizQuestion, QuizAttempt
│       │   └── progress.py         # ProgressEvent (append-only log)
│       ├── schemas/                # Pydantic request/response schemas
│       │   ├── document.py
│       │   ├── chat.py
│       │   ├── plan.py
│       │   ├── quiz.py
│       │   └── progress.py
│       ├── api/v1/
│       │   ├── router.py           # Aggregates all sub-routers
│       │   ├── knowledge.py        # Upload, list, delete documents
│       │   ├── chat.py             # Sessions CRUD + SSE streaming messages
│       │   ├── plans.py            # Generate plans, CRUD, topic status updates
│       │   ├── quizzes.py          # Generate quizzes, CRUD, submit attempts
│       │   └── progress.py         # Summary stats, event log
│       └── services/
│           ├── ingestion/
│           │   ├── chunker.py      # Heading-aware MD chunker + PDF page chunker
│           │   ├── markdown_loader.py
│           │   └── pdf_loader.py   # pdfplumber + table extraction
│           ├── rag/
│           │   ├── embedder.py     # SentenceTransformer singleton (@lru_cache)
│           │   ├── qdrant_store.py # Upsert dense+sparse vectors; delete by document
│           │   └── retriever.py    # Hybrid search: Prefetch dense+sparse → RRF fusion
│           └── agents/
│               ├── llm_factory.py  # get_llm() — only provider logic in the codebase
│               ├── mentor_agent.py # LangGraph: retrieve → generate → extract_citations
│               └── planner_agent.py # LangGraph: summarize_kb → analyze_gaps → generate_plan → persist
│
└── frontend/
    ├── Dockerfile                  # Multi-stage: build → standalone runner
    ├── package.json
    ├── next.config.ts              # output: standalone
    ├── tailwind.config.ts
    ├── tsconfig.json
    └── src/
        ├── app/
        │   ├── layout.tsx          # Root layout with sidebar
        │   ├── page.tsx            # redirect → /chat
        │   ├── chat/
        │   │   ├── page.tsx        # Session list + create new
        │   │   └── [id]/page.tsx   # Chat with streaming, citations
        │   ├── knowledge/
        │   │   └── page.tsx        # Upload + document list with status polling
        │   ├── plans/
        │   │   ├── page.tsx        # Plans list + generate modal
        │   │   └── [id]/page.tsx   # Topic tree with progress tracking
        │   ├── quizzes/
        │   │   ├── page.tsx        # Quiz list + generate modal
        │   │   └── [id]/page.tsx   # Quiz taking with scoring
        │   └── progress/
        │       └── page.tsx        # Stats dashboard
        └── lib/
            ├── api.ts              # Typed API client (fetch wrapper)
            ├── streaming.ts        # SSE ReadableStream async generator
            ├── types.ts            # TypeScript interfaces
            └── utils.ts            # cn(), formatDate(), formatFileSize()
```

---

## Database Schema (PostgreSQL)

### Core Tables

**`documents`** — uploaded files
- `id UUID PK`, `filename`, `original_filename`, `file_type` (markdown|pdf)
- `file_size`, `status` (pending|processing|ready|error), `chunk_count`, `error_msg`

**`document_chunks`** — parsed and chunked content
- `id UUID PK`, `document_id FK`, `qdrant_id UUID UNIQUE` (matches Qdrant point ID)
- `chunk_index`, `content`, `h1_title`, `h2_title`, `h3_title`, `page_number`, `token_count`

**`chat_sessions`** — conversation containers
- `id UUID PK`, `title` (auto-set from first message)

**`chat_messages`** — individual messages
- `id UUID PK`, `session_id FK`, `role` (user|assistant)
- `content`, `source_chunks UUID[]` (cited chunk IDs)

**`learning_plans`** — AI-generated learning plans
- `id UUID PK`, `title`, `description`, `goal`, `status` (active|completed)
- `raw_analysis JSONB` (gap analysis + KB summary)

**`plan_topics`** — topics with self-referential tree
- `id UUID PK`, `plan_id FK`, `parent_id FK → plan_topics` (subtopics)
- `title`, `description`, `order_index`, `status` (not_started|in_progress|completed)
- `estimated_hours`

**`quizzes`** + **`quiz_questions`** + **`quiz_attempts`**
- Questions have `question_type` (multiple_choice|true_false|short_answer)
- `options JSONB` ({"A": "...", "B": "..."})
- Attempts store `score` (0–1) and per-question `answers JSONB`

**`progress_events`** — append-only event log
- `event_type`, `entity_id`, `entity_type`, `metadata JSONB`

---

## Qdrant Vector Store

**Collection:** `knowledge_chunks`

```
vectors_config:
  dense:  size=384, distance=COSINE     # all-MiniLM-L6-v2
sparse_vectors_config:
  sparse: on_disk=false                  # BM25-style tokens

payload per point:
  chunk_id, document_id, filename, file_type
  content, h1_title, h2_title, h3_title
  page_number, chunk_index, token_count
```

**Hybrid search query:**
```
Prefetch dense (top-15) + Prefetch sparse (top-15) → FusionQuery(RRF) → top-5
```

---

## LangGraph Agents

### Mentor Agent (`mentor_agent.py`)
```
retrieve_context → generate_response → END
```
- **retrieve**: `hybrid_search(user_message, top_k=5)`
- **generate**: builds context string with numbered citations → `llm.ainvoke([system, *history, human])` → extracts `[1][2]` citations
- Compiled once at startup, stored in `app.state.mentor_graph`

### Planner Agent (`planner_agent.py`)
```
summarize_knowledge → analyze_gaps → generate_plan ──(retry loop)──► persist_plan → END
                                           ↑___________________________|
```
- **summarize**: scroll Qdrant for unique filenames/headings → build topic list
- **analyze**: LLM compares KB topics vs learning goal → gap analysis text
- **generate**: LLM produces structured JSON plan; retries up to 3× on JSONDecodeError
- **persist**: writes `LearningPlan` + `PlanTopic` tree to PostgreSQL

### LLM Factory (`llm_factory.py`)
```python
def get_llm(streaming=False) -> BaseChatModel:
    if settings.llm_provider == "anthropic":
        return ChatAnthropic(model=settings.anthropic_model, ...)
    return ChatOpenAI(model=settings.openai_model, ...)
```
Single function — all provider logic lives here.

---

## API Routes

```
# Knowledge
POST   /api/v1/knowledge/upload                  → 202, starts background processing
GET    /api/v1/knowledge/documents               → list with status
GET    /api/v1/knowledge/documents/{id}          → single document
GET    /api/v1/knowledge/documents/{id}/chunks   → chunk list
DELETE /api/v1/knowledge/documents/{id}          → delete doc + Qdrant points + file

# Chat
POST   /api/v1/chat/sessions                     → create session
GET    /api/v1/chat/sessions                     → list sessions
GET    /api/v1/chat/sessions/{id}                → session + messages
DELETE /api/v1/chat/sessions/{id}
POST   /api/v1/chat/sessions/{id}/messages       → SSE stream

# SSE stream format:
# data: {"type":"token","text":"A "}
# data: {"type":"done","citations":[{"chunk_id":"...","filename":"...","heading":"..."}]}

# Plans
POST   /api/v1/plans/generate                    → triggers planner agent, returns full plan
GET    /api/v1/plans                             → list
GET    /api/v1/plans/{id}                        → plan + topic tree
PATCH  /api/v1/plans/{id}/topics/{tid}           → update topic status

# Quizzes
POST   /api/v1/quizzes/generate                  → LLM generates questions from content
GET    /api/v1/quizzes                           → list
GET    /api/v1/quizzes/{id}                      → quiz + questions (no correct answers)
POST   /api/v1/quizzes/{id}/attempts             → submit answers → score + explanations

# Progress
GET    /api/v1/progress/summary                  → aggregate stats
GET    /api/v1/progress/events                   → recent event log
```

---

## Document Processing Pipeline

```
User uploads file
       │
       ▼
POST /upload → save file → Document(status=pending) → 202 response
                                    │
                         BackgroundTask starts
                                    │
                         status = "processing"
                                    │
                    ┌───────────────┴──────────────┐
               markdown?                          pdf?
                    │                              │
          load_markdown()                      load_pdf()
         heading-aware chunks              pdfplumber pages
                    │                              │
                    └───────────────┬──────────────┘
                                    │
                              chunk list
                                    │
                    ┌───────────────┴──────────────┐
                    │                              │
             PostgreSQL                         Qdrant
         document_chunks rows              upsert dense+sparse vectors
                    │                              │
                    └───────────────┬──────────────┘
                                    │
                         status = "ready"
                         chunk_count = N
```

---

## Frontend Pages

| Page | Route | Description |
|---|---|---|
| Chat list | `/chat` | Session list, create new chat |
| Chat session | `/chat/[id]` | Streaming message interface with source citations |
| Knowledge | `/knowledge` | Upload dropzone + document list with status polling |
| Plans | `/plans` | Plan list + generate-plan modal |
| Plan detail | `/plans/[id]` | Topic tree with progress, collapsible subtopics |
| Quizzes | `/quizzes` | Quiz list + generate-quiz modal |
| Quiz | `/quizzes/[id]` | Question-by-question with immediate scoring |
| Progress | `/progress` | Stats dashboard (docs, topics, quiz scores) |

---

## Implementation Phases

### Phase 1 — Infrastructure ✅
- `docker-compose.yml` (postgres:16-alpine, qdrant:v1.9.2, backend, frontend)
- `backend/pyproject.toml` (uv)
- `core/config.py`, `core/database.py`, `main.py`
- SQLAlchemy models + Alembic migration `0001_initial.py`

### Phase 2 — Document Ingestion Pipeline ✅
- `services/ingestion/chunker.py` — heading-aware markdown + PDF chunking
- `services/ingestion/markdown_loader.py`, `pdf_loader.py`
- `services/rag/embedder.py` — SentenceTransformer `@lru_cache` singleton
- `services/rag/qdrant_store.py` — Qdrant collection init + upsert dense+sparse
- `api/v1/knowledge.py` — upload → `BackgroundTask` → poll `status`

### Phase 3 — RAG + AI Agents ✅
- `services/rag/retriever.py` — RRF hybrid search
- `services/agents/llm_factory.py`
- `services/agents/mentor_agent.py` — LangGraph mentor
- `services/agents/planner_agent.py` — LangGraph planner
- `api/v1/chat.py` — SSE streaming
- `api/v1/plans.py`, `api/v1/quizzes.py`, `api/v1/progress.py`

### Phase 4 — Frontend ✅
- Next.js 15 + Tailwind CSS
- `lib/api.ts` (typed fetch wrapper), `lib/streaming.ts` (SSE async generator)
- All 8 pages with full functionality

### Phase 5 — Upcoming
- [ ] Alembic `__init__.py` in `versions/`
- [ ] Unit tests for chunker and RRF retriever
- [ ] GitHub repository import (clone → process markdown files)
- [ ] User authentication (JWT or session)
- [ ] Concept graph visualization (topics connected across documents)
- [ ] Real token-level streaming (provider-native, not word-by-word)
- [ ] Export progress as PDF report

### Phase 6 — Advanced RAG & Agentic Techniques (primary technical track)

The chosen learning focus: high-tech RAG/agent instruments, each measured by an eval
number. Three tracks, run in order — A is the foundation everything else is measured
against. See `NEXT_STEPS.md` §1–3 for task-level detail.

**Track A — Eval & observability (foundation, do first)**
- [ ] Retrieval eval harness: 15–30 (query → gold chunk) pairs → Recall@k, MRR, nDCG
- [ ] RAGAS: faithfulness, answer_relevancy, context_precision/recall (LLM-as-judge)
- [ ] LangSmith tracing (`LANGCHAIN_TRACING_V2`) — per-node graph traces, tokens, latency

**Track C — Advanced retrieval (after A; one experiment at a time, measure each)**
- [ ] HyDE — embed a hypothetical answer, not the bare question
- [ ] Smart chunking — recursive / late chunking / semantic chunking (`chunker.py`)
- [ ] Parent-document retrieval — search small chunks, feed parent/neighbor blocks to LLM
- [ ] MMR de-dup on top-k; metadata filter on `document_id` (also enables §7-A)
- [ ] Late-interaction (ColBERT/ColPali) via Qdrant multivector — SOTA, measure the cost
- [ ] Optional: swap embeddings (bge/e5/gte) — dimension change → recreate collection

**Track B — Agentic RAG (after A+C)**
- [ ] Tool-calling mentor (`create_react_agent` or agent⇄tools loop) — LLM owns retrieval
- [ ] Query decomposition — split compound questions into sub-queries
- [ ] CRAG (Corrective RAG) — grade retrieved context, rewrite+re-retrieve on miss
- [ ] Self-RAG / reflection loop — critique own draft, re-retrieve if unsupported
- [ ] Measure via RAGAS (faithfulness/relevancy before/after); debug via LangSmith

**New deps:** `ragas`, `langsmith` (+ ColBERT model if Track C late-interaction lands)
**Touched:** `retriever.py`, `qdrant_store.py`, `chunker.py`, `mentor_agent.py`, `lifespan.py`

### Phase 7 — Interactive Learning (application surface)

Turns scattered uploads into **learning units**: collections spanning multiple documents,
section-level quizzes, and exercises with real grading. Builds on existing fields
(`h1/h2/h3_title` on chunks, `source_chunk_id` on questions, Qdrant payload filters) —
mostly new endpoints + one migration, not a rewrite. See `NEXT_STEPS.md` §6 for the
task-level breakdown and effort labels.

**Naming:** the new document-grouping entity is `Collection` in code/DB (UI label: "topic"),
to avoid colliding with the existing `PlanTopic` (a node in a plan tree). A document section
is identified by `(document_id, heading_path)`, not a bare heading string (headings repeat
across documents).

**Sequencing:** A and B change retrieval (doc filter, section quiz) → measure quality
*after* the eval harness (NEXT_STEPS §1). C's "explain in your own words" reuses the
tool-calling mentor (NEXT_STEPS §2) → do it after.

**A. Collections as document groups**
- [ ] New `collections` table + `collection_documents` (many-to-many) — Alembic migration
- [ ] `hybrid_search(..., document_ids=[...])` → Qdrant `Filter` on `document_id` payload
- [ ] Collection-focused chat: `collection_id` on session → mentor answers only within its docs
- [ ] `/collections` UI: create, attach documents, "Plan / Quiz / Ask" actions

**B. Section-level quizzes & material**
- [ ] `GET /knowledge/documents/{id}/outline` — table of contents from chunk headings
- [ ] `GenerateQuizRequest` accepts `(document_id, heading_path)` → filter chunks → focused quiz
- [ ] Populate `QuizQuestion.source_chunk_id` → clickable source citation per question

**C. Interactive exercises**
- [ ] LLM grading for `short_answer` (correct/partial/wrong + feedback) — replaces `==` compare
- [ ] Fill-in-the-blank question type
- [ ] "Explain in your own words" → Socratic follow-ups via mentor graph (after §2)
- [ ] (optional) Flashcards + spaced repetition (`review_schedule` table) — do last

**D. Adaptivity & progress**
- [ ] Coverage map: link quiz attempts to section/collection → "10/14 sections, avg 78%"
- [ ] Weak-spot detection: mentor suggests revisiting low-score sections

**New tables:** `collections`, `collection_documents`, (optional) `review_schedule`
**Touched:** `retriever.py`, `qdrant_store.py`, `quizzes.py`, `chat.py`, `progress.py`

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | postgresql+asyncpg://... | PostgreSQL async connection string |
| `QDRANT_URL` | http://localhost:6333 | Qdrant instance URL |
| `LLM_PROVIDER` | anthropic | `anthropic` or `openai` |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `ANTHROPIC_MODEL` | claude-sonnet-4-6 | Claude model ID |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `OPENAI_MODEL` | gpt-4o-mini | OpenAI model ID |
| `EMBEDDING_MODEL` | sentence-transformers/all-MiniLM-L6-v2 | HuggingFace model name |
| `EMBEDDING_DIMENSION` | 384 | Must match the embedding model output |
| `RETRIEVAL_TOP_K` | 5 | Number of chunks to retrieve per query |
| `CHUNK_SIZE` | 512 | Max tokens per chunk |
| `CHUNK_OVERLAP` | 64 | Token overlap between adjacent chunks |

---

## Verification Checklist

1. `cp .env.example .env` → fill `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`
2. `docker compose up -d` → all 4 services healthy
3. `curl http://localhost:8000/health` → `{"status":"ok"}`
4. Open `http://localhost:6333/dashboard` → `knowledge_chunks` collection exists
5. Upload `test.md` → poll `GET /api/v1/knowledge/documents/{id}` → `status: ready`
6. `POST /api/v1/chat/sessions` → send message → SSE stream returns tokens + citations
7. `POST /api/v1/plans/generate` → `{"goal": "Master FastAPI"}` → plan with topics
8. Open `http://localhost:3000` → full UI works end-to-end
