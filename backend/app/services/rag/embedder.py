"""Singleton SentenceTransformer embedder."""
import asyncio
import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    logger.info("Loading embedding model: %s", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded.")
    return model


def embed_texts(texts: list[str], prefix: str = "") -> list[list[float]]:
    """Synchronous encode. CPU-bound — do NOT call directly from async code.

    ``prefix`` is prepended to every text before encoding. e5 models require a
    "query: " / "passage: " instruction prefix (see config); prefix-free models
    pass "".
    """
    model = get_embedder()
    if prefix:
        texts = [prefix + t for t in texts]
    # normalize_embeddings=True → unit-length vectors, so cosine == dot product,
    # which is what the Qdrant COSINE distance expects.
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed documents/chunks for storage (e5 "passage: " prefix)."""
    return embed_texts(texts, settings.embedding_passage_prefix)


def embed_query(query: str) -> list[float]:
    """Embed a search query (e5 "query: " prefix).

    Asymmetric prefixing matters: a query and the passage that answers it are
    encoded with *different* prefixes, which is how e5 was trained.
    """
    return embed_texts([query], settings.embedding_query_prefix)[0]


# --- Late chunking (Track C, exp.5) ----------------------------------------
#
# Normal ingest embeds each chunk in isolation: the chunk's vector only ever
# sees the chunk's own tokens. Late chunking inverts the order — embed the whole
# document FIRST so every token is contextualized by the entire document, THEN
# pool the token embeddings over each chunk's span. A chunk-vector then "knows"
# the document around it (a pronoun resolves, a bare definition inherits its
# subject) without re-splitting finer (the failure mode of semantic chunking,
# Module 9). Idea: Günther et al., "Late Chunking" (Jina AI, 2024).
#
# The hard constraint here: e5's encoder reads at most ``max_seq_length`` (512)
# tokens, so a long document CANNOT go through in one window — the literal
# premise of late chunking is unavailable on a short-context encoder. We
# approximate it with a token-level SLIDING WINDOW (passage-prefixed, overlap
# averaged on the seam), so each token's embedding still attends over a 512-token
# neighbourhood rather than just its own chunk. This is the central, honestly
# acknowledged confound: contextualization is *local* (one window wide), not
# truly global. See LEARNING_NOTES Module 10.


def _transformer():
    """The underlying HF encoder + tokenizer behind the SentenceTransformer.

    Late chunking needs token-level hidden states (``last_hidden_state``), which
    ``SentenceTransformer.encode`` hides behind its pooling layer — so we reach
    the first module (the Transformer) and drive ``auto_model`` directly.
    """
    model = get_embedder()
    first = model[0]
    return model, first.auto_model, first.tokenizer


def embed_spans_late(body: str, spans: list[tuple[int, int]]) -> list[list[float]]:
    """Late-chunking embeddings: one normalized vector per (start, end) char span.

    1. Tokenize the whole ``body`` once, keeping char offsets per token.
    2. Run the encoder over overlapping <=max_seq windows (passage-prefixed),
       averaging token embeddings where windows overlap → a contextualized
       embedding for every document token.
    3. Mean-pool the tokens inside each span and L2-normalize (so cosine == dot,
       matching the Qdrant COSINE distance and ``embed_passages`` outputs).

    Pools the span's *content* tokens only (not the [CLS]/prefix/[SEP] specials):
    that is the late-chunking formulation — the prefix conditions the contextual
    pass, but the chunk vector is the mean of the chunk's own contextualized
    tokens. A minor, deliberate asymmetry vs naive ``embed_passages`` (which means
    over the specials too), noted in LEARNING_NOTES Module 10.
    """
    import torch

    if not body.strip() or not spans:
        return []

    model, auto_model, tokenizer = _transformer()
    prefix = settings.embedding_passage_prefix
    prefix_ids = (
        tokenizer(prefix, add_special_tokens=False)["input_ids"] if prefix else []
    )

    enc = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    if not ids:
        return [[] for _ in spans]

    model_max = getattr(model, "max_seq_length", None) or 512
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    n_special = (cls_id is not None) + (sep_id is not None)
    # window = doc tokens that fit alongside the prefix and the [CLS]/[SEP] pair
    window = max(16, model_max - len(prefix_ids) - n_special)
    overlap = min(settings.late_chunk_window_overlap, window // 2)
    stride = max(1, window - overlap)

    dim = auto_model.config.hidden_size
    sum_emb = torch.zeros(len(ids), dim)
    count = torch.zeros(len(ids))
    head = (1 if cls_id is not None else 0) + len(prefix_ids)

    auto_model.eval()
    with torch.no_grad():
        for start in range(0, len(ids), stride):
            win = ids[start : start + window]
            if not win:
                break
            seq = (
                ([cls_id] if cls_id is not None else [])
                + prefix_ids
                + win
                + ([sep_id] if sep_id is not None else [])
            )
            input_ids = torch.tensor([seq])
            attn = torch.ones_like(input_ids)
            hidden = auto_model(input_ids=input_ids, attention_mask=attn).last_hidden_state[0]
            win_emb = hidden[head : head + len(win)]
            sum_emb[start : start + len(win)] += win_emb
            count[start : start + len(win)] += 1.0
            if start + window >= len(ids):
                break

    token_emb = sum_emb / count.clamp(min=1.0).unsqueeze(1)

    vectors: list[list[float]] = []
    for cs, ce in spans:
        idx = [i for i, (os, oe) in enumerate(offsets) if oe > cs and os < ce and oe > os]
        if not idx:  # span fell between tokens — pin to the nearest token start
            idx = [min(range(len(offsets)), key=lambda i: abs(offsets[i][0] - cs))]
        pooled = token_emb[idx].mean(dim=0)
        pooled = torch.nn.functional.normalize(pooled, dim=0)
        vectors.append(pooled.tolist())
    return vectors


# Async wrappers: model.encode is a blocking CPU-bound call. Running it directly
# in an async path would freeze the event loop (no other request progresses for
# the whole encode). asyncio.to_thread offloads it to a worker thread so the
# loop stays responsive.
async def aembed_passages(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(embed_passages, texts)


async def aembed_query(query: str) -> list[float]:
    return await asyncio.to_thread(embed_query, query)
