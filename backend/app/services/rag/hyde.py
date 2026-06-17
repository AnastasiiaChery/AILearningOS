"""HyDE — Hypothetical Document Embeddings (Track C, exp.2).

A short raw question ("что такое RRF?") sits awkwardly in embedding space: it is
phrased like a *question*, while the passages that answer it are phrased like
*statements*. HyDE bridges that gap — an LLM first drafts a hypothetical answer,
and we search by the embedding of that fabricated passage. Even if the draft is
factually shaky, its vector lands in "answer space", nearer the real passages.

Only the dense side consumes the hypothetical doc (see retriever.py); the sparse
BM25 side keeps the original query, since a long generated paragraph would dilute
the exact keywords sparse retrieval relies on.

Caveat worth measuring (not assuming): HyDE pays off most when the encoder is weak
or the query is terse/vague. With a strong multilingual encoder already near the
recall ceiling (exp.1), the hypo doc may add drift instead of signal — that's an
empirical question the eval harness answers, not a given.
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.services.agents.llm_factory import get_llm

logger = logging.getLogger(__name__)

# Keep the draft short and answer-shaped. We ask for the *answer's language* to
# match the question so a Ukrainian query produces a Ukrainian hypo doc (the
# corpus is Ukrainian; an English draft would land in the wrong language cluster
# of the multilingual encoder).
HYDE_PROMPT = """Write a short, factual passage (2-4 sentences) that would directly \
answer the question below, as if it were an excerpt from a reference document. \
Write in the SAME language as the question. State facts plainly — do not hedge, \
do not say "I don't know", do not add preamble or restate the question. If unsure \
of specifics, write a plausible passage on the topic anyway.

Question: {question}

Passage:"""


async def generate_hypothetical(query: str) -> str:
    """Return an LLM-drafted hypothetical answer passage for ``query``.

    Falls back to the original query on any failure or empty output, so HyDE can
    never make retrieval *worse than no-HyDE* due to an LLM hiccup — the dense
    side just degrades gracefully to the plain query.
    """
    try:
        llm = get_llm(streaming=False, temperature=0)
        resp = await llm.ainvoke([HumanMessage(content=HYDE_PROMPT.format(question=query))])
        doc = (resp.content or "").strip()
        if not doc:
            logger.warning("HyDE produced empty output for %r; falling back to query.", query)
            return query
        logger.debug("HyDE for %r → %r", query, doc[:120])
        return doc
    except Exception:  # noqa: BLE001 — never let HyDE break the search path
        logger.exception("HyDE generation failed for %r; falling back to query.", query)
        return query
