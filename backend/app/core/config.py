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
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # RAG
    retrieval_top_k: int = 5
    # Max tokens per chunk. Hard-capped in the chunker to the embedding model's
    # max_seq_length (all-MiniLM-L6-v2 = 256). Lower this for sharper, more
    # granular retrieval; raise (up to 256) for more context per chunk.
    chunk_size: int = 256
    chunk_overlap: int = 48

    # Reranking (stage 2): pull N candidates from hybrid search, cross-encode,
    # keep retrieval_top_k. Set rerank_enabled=False to compare against raw RRF.
    rerank_enabled: bool = True
    rerank_candidates: int = 20
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # App
    upload_dir: str = "/app/uploads"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    log_level: str = "info"
    debug: bool = False


settings = Settings()
