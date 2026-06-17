from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = "postgresql+asyncpg://app:devpassword@localhost:5432/ailearningos"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "knowledge_chunks"

    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Embeddings
    # Multilingual (Track C, exp.1): the corpus is Ukrainian, so an English-only
    # encoder (all-MiniLM-L6-v2) gave weak dense recall. e5-base is multilingual,
    # 768d, max_seq_length 512. Changing the model/dimension requires recreating
    # the Qdrant collection + re-ingest (see app/eval/reingest.py).
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dimension: int = 768
    # e5 was trained with instruction prefixes: queries must be prefixed with
    # "query: " and passages with "passage: ". Skipping them measurably degrades
    # retrieval. For prefix-free models (e.g. MiniLM) set both to "".
    embedding_query_prefix: str = "query: "
    embedding_passage_prefix: str = "passage: "

    # RAG
    retrieval_top_k: int = 5
    # Max tokens per chunk. Hard-capped in the chunker to the embedding model's
    # max_seq_length (e5-base = 512; clamped further by this value). Lower this
    # for sharper, more granular retrieval; raise for more context per chunk.
    # Kept at 256 so chunk granularity matches the MiniLM baseline (only the
    # encoder changes) — the effective cap is min(chunk_size, max_seq)-16.
    chunk_size: int = 256
    chunk_overlap: int = 48

    # Chunking strategy (Track C, exp.3): how a section's text is cut.
    #   "recursive" – split at natural boundaries (paragraph → line → sentence),
    #                 greedily packed to the token budget with whole-unit
    #                 overlap. Never cuts a phrase mid-sentence unless one
    #                 sentence alone exceeds the model limit. Default.
    #   "token"     – legacy sliding window over raw token ids (Modules 1–7
    #                 baseline). Kept for reproducible before/after comparison.
    chunking_strategy: str = "recursive"

    # HyDE (Track C, exp.2): Hypothetical Document Embeddings. Before searching,
    # an LLM writes a hypothetical answer to the query; we embed THAT for the
    # dense side instead of the short raw question — a fabricated passage lives
    # closer to real passages in vector space than a terse query does. Only the
    # dense side uses it; sparse/BM25 keeps the original query (the hypo doc would
    # just add keyword noise). Off by default — it's an experiment; flip on only
    # once the eval harness shows it helps on this corpus.
    hyde_enabled: bool = False

    # Reranking (stage 2): pull N candidates from hybrid search, cross-encode,
    # keep retrieval_top_k. Set rerank_enabled=False to compare against raw RRF.
    # mmarco-mMiniLMv2 is a multilingual cross-encoder (vs the English-only
    # ms-marco-MiniLM, which actively hurt on the Ukrainian corpus — see Module 5).
    rerank_enabled: bool = True
    rerank_candidates: int = 20
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # App
    upload_dir: str = "/app/uploads"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    log_level: str = "info"
    debug: bool = False


settings = Settings()
