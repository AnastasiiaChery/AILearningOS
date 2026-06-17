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

    chunks: list[Chunk] = []
    for sec in sections:
        raw = sec["text"]
        for piece in _split_by_tokens(raw, max_tok, overlap_tok):
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
    chunks: list[Chunk] = []

    for page_num, page_text in pages:
        page_text = page_text.strip()
        if not page_text:
            continue
        for piece in _split_by_tokens(page_text, max_tok, overlap_tok):
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
