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

    # Chunking strategy (Track C, exp.3/4): how a section's text is cut.
    #   "recursive" – split at natural boundaries (paragraph → line → sentence),
    #                 greedily packed to the token budget with whole-unit
    #                 overlap. Never cuts a phrase mid-sentence unless one
    #                 sentence alone exceeds the model limit. Default.
    #   "semantic"  – (exp.4) cut where meaning jumps: embed every sentence,
    #                 measure cosine distance between neighbours, place a boundary
    #                 at the top-percentile jumps. Reuses the embedder singleton;
    #                 heavier at ingest (one extra encode per sentence). Token cap
    #                 still enforced by repacking over-long segments.
    #   "late"      – (exp.5) late chunking: cut at natural boundaries (same as
    #                 recursive, no overlap) but embed the WHOLE document first and
    #                 mean-pool token embeddings over each chunk's span, so every
    #                 chunk-vector is contextualized by the surrounding document.
    #                 Bounded by the encoder's 512-token window → a token-level
    #                 sliding window approximates the single pass (see embedder.py).
    #   "token"     – legacy sliding window over raw token ids (Modules 1–7
    #                 baseline). Kept for reproducible before/after comparison.
    chunking_strategy: str = "recursive"
    # Semantic chunking breakpoint percentile (only used when strategy="semantic").
    # A consecutive-sentence distance at/above this percentile starts a new chunk.
    # Higher → fewer, larger chunks (only the sharpest topic shifts cut); lower →
    # more, smaller chunks. 95 keeps roughly the top ~5% of jumps as boundaries.
    semantic_breakpoint_percentile: float = 95.0
    # Late chunking (strategy="late"): token overlap between consecutive encoder
    # windows. The overlap is averaged so the contextualized embeddings join
    # smoothly across the 512-token window seam instead of jumping discontinuously.
    late_chunk_window_overlap: int = 64

    # HyDE (Track C, exp.2): Hypothetical Document Embeddings. Before searching,
    # an LLM writes a hypothetical answer to the query; we embed THAT for the
    # dense side instead of the short raw question — a fabricated passage lives
    # closer to real passages in vector space than a terse query does. Only the
    # dense side uses it; sparse/BM25 keeps the original query (the hypo doc would
    # just add keyword noise). Off by default — it's an experiment; flip on only
    # once the eval harness shows it helps on this corpus.
    hyde_enabled: bool = False

    # Parent-document retrieval (Track C, exp.6): search the small chunks
    # (precision), but on the way back expand each hit into its parent context
    # (recall/completeness for the LLM). Two shapes:
    #   "section" – gather every chunk sharing the hit's heading-path (same
    #               document_id + h1/h2/h3) and stitch them by chunk_index.
    #   "window"  – the hit's chunk_index ± parent_window within the same document.
    # Hits that map to the same parent are de-duplicated to a single block (which
    # frees top-k slots for other parents). The index is untouched — expansion is
    # purely a return-path transform, so no re-ingest. Off by default: measure
    # first, and most of the payoff (fuller LLM context) is invisible to the
    # retrieval harness — see LEARNING_NOTES Module 11.
    parent_retrieval: str = "off"  # off | section | window
    parent_window: int = 1

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
