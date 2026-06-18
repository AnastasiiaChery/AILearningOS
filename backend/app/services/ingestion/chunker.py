"""Heading-aware document chunker (token-accurate).

Token counting and splitting use the *embedding model's own tokenizer*.
Rationale: ``all-MiniLM-L6-v2`` silently truncates input beyond its
``max_seq_length`` (256 tokens). That limit is defined in the model's own
WordPiece tokens — so counting with anything else (chars//4, tiktoken's BPE)
would let oversized chunks slip through and lose ~90% of their text at embed
time. Counting with the real tokenizer guarantees every chunk fits.
"""
import re
from dataclasses import dataclass

from app.core.config import settings


@dataclass
class Chunk:
    content: str
    chunk_index: int
    h1_title: str | None = None
    h2_title: str | None = None
    h3_title: str | None = None
    page_number: int | None = None
    token_count: int = 0
    # Pre-computed dense vector. Set only by late chunking (Track C, exp.5), where
    # the vector must be pooled from the whole-document embedding pass and cannot
    # be recomputed from the chunk text alone. None → upsert embeds the text
    # normally (every other strategy).
    dense_vector: list[float] | None = None


def _tokenizer():
    """The embedding model's tokenizer (loaded once via the embedder singleton)."""
    from app.services.rag.embedder import get_embedder

    return get_embedder().tokenizer


def _effective_max_tokens() -> int:
    """Largest chunk we allow, never exceeding what the model can actually read.

    Clamp the configured ``chunk_size`` to the model's ``max_seq_length`` and
    leave 2 tokens of headroom for the [CLS]/[SEP] specials added at encode time.
    """
    from app.services.rag.embedder import get_embedder

    model = get_embedder()
    model_max = getattr(model, "max_seq_length", None) or 256
    # Headroom (16) absorbs the [CLS]/[SEP] specials AND the tokenizer
    # decode->encode round-trip drift: a window of N ids, decoded to text and
    # re-encoded at embed time, can come back as a few more tokens.
    return max(16, min(settings.chunk_size, model_max) - 16)


def count_tokens(text: str) -> int:
    """Real token count (no special tokens) using the model's tokenizer."""
    return len(_tokenizer().encode(text, add_special_tokens=False))


def _split_by_tokens(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Sliding-window split over *actual* tokens with token-level overlap.

    Tokenize once, slide a window of ``max_tokens`` ids stepping by
    ``max_tokens - overlap_tokens``, then decode each window back to text.
    Every returned chunk is guaranteed to be <= max_tokens tokens.
    """
    tok = _tokenizer()
    ids = tok.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text]

    step = max(1, max_tokens - overlap_tokens)
    chunks: list[str] = []
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            break
        piece = tok.decode(window).strip()
        if piece:
            chunks.append(piece)
        if start + max_tokens >= len(ids):
            break
    return chunks or [text]


# --- Recursive boundary-aware splitting (Track C, exp.3) -------------------
#
# The token sliding window above is blunt: it cuts at arbitrary token offsets,
# slicing through the middle of a sentence — even a word. A query's answer can
# end up straddling two chunks, with neither holding the whole thought. The
# recursive splitter instead cuts at the *coarsest natural boundary that fits*
# (blank line → newline → sentence end), then greedily packs whole units up to
# the token budget. A phrase is split only when a single sentence alone exceeds
# the model limit, where the token window is the sole safe fallback.

# Sentence boundary: a sentence-final mark followed by whitespace. Good enough
# for our prose corpus (UK/EN); abbreviations may over-split, which is harmless
# — units are only re-packed, never dropped.
_SENTENCE_RE = r"(?<=[.!?…])\s+"
# Recursion order, coarsest first.
_SEPARATORS = (r"\n\s*\n", r"\n", _SENTENCE_RE)


def _atomize(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Break text into units that each fit in ``max_tokens``.

    Try separators coarsest-first; a separator only "fires" if it actually
    splits the text into more than one non-empty part, otherwise fall through
    to the next finer one. A blob with no usable boundary that is still too
    long is cut by the token window (the only guarantee of the hard cap).
    Every returned unit is <= max_tokens.
    """
    text = text.strip()
    if not text:
        return []
    if count_tokens(text) <= max_tokens:
        return [text]
    for pattern in _SEPARATORS:
        parts = [p for p in re.split(pattern, text) if p.strip()]
        if len(parts) > 1:
            units: list[str] = []
            for p in parts:
                units.extend(_atomize(p, max_tokens, overlap_tokens))
            return units
    return _split_by_tokens(text, max_tokens, overlap_tokens)


def _pack(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack units into chunks <= max_tokens.

    Unlike token-window overlap (which cuts mid-phrase), the carried overlap is
    whole trailing units summing to <= overlap_tokens — so chunk boundaries and
    their overlap both land on sentence/paragraph edges. Units are joined with a
    single space: re-tokenizing the joined string then stays <= the per-unit
    token sum (SentencePiece merges the leading space), so the model limit holds.
    """
    counted = [(u, count_tokens(u)) for u in units]
    chunks: list[str] = []
    cur: list[tuple[str, int]] = []
    cur_tok = 0
    for u, ut in counted:
        if cur and cur_tok + ut > max_tokens:
            chunks.append(" ".join(t for t, _ in cur))
            # seed the next chunk with trailing units for continuity, capped so
            # the carry plus the incoming unit still fits the budget.
            budget = min(overlap_tokens, max_tokens - ut)
            carry: list[tuple[str, int]] = []
            carry_tok = 0
            for prev in reversed(cur):
                if carry_tok + prev[1] > budget:
                    break
                carry.insert(0, prev)
                carry_tok += prev[1]
            cur = carry
            cur_tok = carry_tok
        cur.append((u, ut))
        cur_tok += ut
    if cur:
        chunks.append(" ".join(t for t, _ in cur))
    return chunks


# --- Semantic boundary splitting (Track C, exp.4) --------------------------
#
# Recursive cuts on *syntactic* edges (paragraph/line/sentence) and packs
# greedily to the token budget — boundaries fall wherever the budget runs out,
# blind to meaning. Semantic chunking instead cuts where the *meaning* jumps:
# embed each sentence, measure cosine distance between consecutive sentences,
# and place a boundary at the largest distances (top (100 − P) percentile). A
# run of topically-coherent sentences stays together; the cut lands on the seam
# between two topics. The token cap still holds — an over-long semantic segment
# is repacked by the recursive packer (the only hard guarantee of the limit).


# Finest sentence-level split for semantic chunking: a sentence-final mark
# followed by whitespace, OR any run of newlines. Unlike ``_atomize`` (which
# only splits when a blob is over-budget), this ALWAYS atomizes to sentences —
# semantic chunking must compare adjacent sentences even when the whole section
# fits the token budget.
_UNIT_SPLIT_RE = r"(?<=[.!?…])\s+|\n+"


def _sentence_units(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split into sentence-sized units, each guaranteed <= max_tokens.

    A single sentence longer than the model limit (rare) is cut by the token
    window — the only hard guarantee of the cap.
    """
    text = text.strip()
    if not text:
        return []
    units: list[str] = []
    for p in re.split(_UNIT_SPLIT_RE, text):
        p = p.strip()
        if not p:
            continue
        if count_tokens(p) <= max_tokens:
            units.append(p)
        else:
            units.extend(_split_by_tokens(p, max_tokens, overlap_tokens))
    return units


def _percentile(values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-free). ``p`` in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _semantic_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Cut at semantic jumps, then honor the token cap.

    1. Split into sentence-sized units (each already <= max_tokens).
    2. Embed every unit with the passage prefix via the shared embedder
       singleton (normalized → cosine == dot product).
    3. Distance between consecutive units = 1 − cosine. A boundary falls after
       any unit whose forward distance is in the top (100 − P) percentile.
    4. Each semantic segment is repacked to the token budget (an over-long
       coherent run still splits; that's the only hard cap guarantee).
    """
    units = _sentence_units(text, max_tokens, overlap_tokens)
    if len(units) <= 1:
        return units

    from app.services.rag.embedder import embed_passages

    embs = embed_passages(units)
    # cosine distance between neighbours (embeddings are unit-length → dot == cos)
    dists = [
        1.0 - sum(a * b for a, b in zip(embs[i], embs[i + 1]))
        for i in range(len(embs) - 1)
    ]
    # threshold at the configured percentile; ``>=`` guarantees at least the
    # single largest jump becomes a boundary, so a section always gets cut where
    # its meaning shifts most (never collapses back to one giant chunk).
    threshold = _percentile(dists, settings.semantic_breakpoint_percentile)

    segments: list[list[str]] = []
    cur: list[str] = [units[0]]
    for i, d in enumerate(dists):
        if d >= threshold:
            segments.append(cur)
            cur = [units[i + 1]]
        else:
            cur.append(units[i + 1])
    segments.append(cur)

    chunks: list[str] = []
    for seg in segments:
        # repack each segment so the hard token cap holds; no cross-segment
        # overlap — the whole point is a clean seam between topics.
        chunks.extend(_pack(seg, max_tokens, overlap_tokens))
    return chunks


# --- Late chunking (Track C, exp.5) ----------------------------------------
#
# Every strategy above produces chunk *text*, which upsert then embeds in
# isolation. Late chunking decouples the boundary from the embedding: it cuts at
# the same natural boundaries as ``recursive`` (no overlap — a clean seam), but
# the chunk vectors are pooled from a single whole-document embedding pass (see
# embedder.embed_spans_late). So the chunker must hand the embedder each chunk's
# *char span* within the document body, not just its text. This block computes
# those spans; the actual contextualized pooling lives in the embedder.


def _unit_spans(text: str, max_tokens: int) -> list[tuple[str, int, int]]:
    """Locate each atomized unit as a verbatim (text, start, end) slice of ``text``.

    Reuses ``_atomize`` (so the unit granularity matches the recursive splitter),
    then walks a cursor to recover each unit's char offset. Units are verbatim
    substrings of ``text`` except the rare token-window fallback (a single
    sentence over the model limit), which is placed at the cursor as a best
    effort — offsets only feed pooling, so a few drifted tokens are harmless.
    """
    spans: list[tuple[str, int, int]] = []
    cursor = 0
    for u in _atomize(text, max_tokens, 0):
        i = text.find(u, cursor)
        if i < 0:
            i = text.find(u)
        if i < 0:
            i = cursor
        spans.append((u, i, i + len(u)))
        cursor = i + len(u)
    return spans


def _pack_spans(unit_spans: list[tuple[str, int, int]], max_tokens: int) -> list[tuple[int, int]]:
    """Greedily pack contiguous units into chunk (start, end) spans <= max_tokens.

    No overlap: late-chunking boundaries are clean seams. The surrounding context
    a chunk needs is injected by the document-level embedding, not by repeating
    neighbouring sentences (that is what makes it leaner than recursive overlap).
    """
    chunks: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    cur_tok = 0
    for u, s, e in unit_spans:
        ut = count_tokens(u)
        if cur_start is not None and cur_tok + ut > max_tokens:
            chunks.append((cur_start, cur_end))
            cur_start, cur_tok = None, 0
        if cur_start is None:
            cur_start = s
        cur_end = e
        cur_tok += ut
    if cur_start is not None:
        chunks.append((cur_start, cur_end))
    return chunks


def _late_chunk_sections(sections: list[dict], max_tokens: int) -> list[Chunk]:
    """Late chunking: pool a whole-document embedding pass over per-chunk spans.

    Concatenate every section's text into one document ``body`` (tracking where
    each section lands), cut each section into contiguous chunk spans, then ask
    the embedder for one contextualized vector per span. The body is the literal
    text the encoder reads — headings stay metadata-only, exactly as in every
    other strategy, so the comparison is apples-to-apples.
    """
    from app.services.rag.embedder import embed_spans_late

    sep = "\n\n"
    parts: list[str] = []
    body_len = 0
    specs: list[dict] = []
    for sec in sections:
        text = (sec.get("text") or "").strip()
        if not text:
            continue
        base = body_len
        parts.append(text)
        body_len += len(text)
        parts.append(sep)
        body_len += len(sep)
        for s, e in _pack_spans(_unit_spans(text, max_tokens), max_tokens):
            specs.append(
                {
                    "h1": sec.get("h1"),
                    "h2": sec.get("h2"),
                    "h3": sec.get("h3"),
                    "page_number": sec.get("page_number"),
                    "start": base + s,
                    "end": base + e,
                }
            )

    body = "".join(parts)
    vectors = embed_spans_late(body, [(c["start"], c["end"]) for c in specs])

    chunks: list[Chunk] = []
    for i, (c, vec) in enumerate(zip(specs, vectors)):
        content = body[c["start"] : c["end"]].strip()
        if not content:
            continue
        chunks.append(
            Chunk(
                content=content,
                chunk_index=len(chunks),
                h1_title=c["h1"],
                h2_title=c["h2"],
                h3_title=c["h3"],
                page_number=c["page_number"],
                token_count=count_tokens(content),
                dense_vector=vec,
            )
        )
    return chunks


def _split_section(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split one section's text into chunks per the configured strategy.

    ``recursive`` cuts on natural boundaries (default); ``semantic`` cuts where
    sentence-to-sentence meaning jumps (embeds every sentence); ``token`` is the
    legacy sliding window kept for reproducible before/after comparison.
    """
    if settings.chunking_strategy == "semantic":
        return _semantic_split(text, max_tokens, overlap_tokens)
    if settings.chunking_strategy == "recursive":
        return _pack(_atomize(text, max_tokens, overlap_tokens), max_tokens, overlap_tokens)
    return _split_by_tokens(text, max_tokens, overlap_tokens)


def chunk_markdown(text: str) -> list[Chunk]:
    """Split markdown text into heading-aware chunks."""
    max_tok = _effective_max_tokens()
    overlap_tok = min(settings.chunk_overlap, max_tok // 2)

    heading_re = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

    # Split document at headings, preserving heading context
    sections: list[dict] = []
    current: dict = {"h1": None, "h2": None, "h3": None, "text": ""}
    last_end = 0

    for m in heading_re.finditer(text):
        before = text[last_end : m.start()].strip()
        if before:
            current["text"] = (current["text"] + "\n\n" + before).strip()
        # Flush current section if it has text
        if current["text"]:
            sections.append(dict(current))
        level = len(m.group(1))
        title = m.group(2).strip()
        if level == 1:
            current = {"h1": title, "h2": None, "h3": None, "text": ""}
        elif level == 2:
            current = {"h1": current["h1"], "h2": title, "h3": None, "text": ""}
        else:
            current = {"h1": current["h1"], "h2": current["h2"], "h3": title, "text": ""}
        last_end = m.end()

    remainder = text[last_end:].strip()
    if remainder:
        current["text"] = (current["text"] + "\n\n" + remainder).strip()
    if current["text"]:
        sections.append(current)

    if settings.chunking_strategy == "late":
        return _late_chunk_sections(sections, max_tok)

    chunks: list[Chunk] = []
    for sec in sections:
        raw = sec["text"]
        for piece in _split_section(raw, max_tok, overlap_tok):
            if piece.strip():
                chunks.append(
                    Chunk(
                        content=piece,
                        chunk_index=len(chunks),
                        h1_title=sec["h1"],
                        h2_title=sec["h2"],
                        h3_title=sec["h3"],
                        token_count=count_tokens(piece),
                    )
                )

    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks


def chunk_pdf_pages(pages: list[tuple[int, str]]) -> list[Chunk]:
    """Chunk PDF pages. pages = [(page_num, text), ...]"""
    max_tok = _effective_max_tokens()
    overlap_tok = min(settings.chunk_overlap, max_tok // 2)

    if settings.chunking_strategy == "late":
        sections = [{"page_number": pn, "text": pt} for pn, pt in pages]
        return _late_chunk_sections(sections, max_tok)

    chunks: list[Chunk] = []

    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue
        for piece in _split_section(page_text, max_tok, overlap_tok):
            if piece.strip():
                chunks.append(
                    Chunk(
                        content=piece,
                        chunk_index=len(chunks),
                        page_number=page_num,
                        token_count=count_tokens(piece),
                    )
                )

    for i, c in enumerate(chunks):
        c.chunk_index = i
    return chunks
