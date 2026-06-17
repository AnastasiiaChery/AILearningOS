"""Resolve golden-set anchors → live chunk_ids.

Why anchors instead of raw chunk_ids: a chunk's UUID is regenerated on every
re-ingestion, and Track C is nothing but re-ingestion (new chunking, parent-doc,
ColBERT…). So the golden set points at chunks by a *stable* anchor — a filename
plus one or more distinctive text snippets — and this module maps those anchors
to whatever chunk_ids currently hold that text in Qdrant. The set never goes
stale; it tracks meaning, not a transient id.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.services.rag.qdrant_store import get_qdrant_client

GOLDSET_PATH = Path(__file__).parent / "goldset.toml"


@dataclass
class GoldQuestion:
    id: str
    file: str
    question: str
    gold: list[str]                 # distinctive snippets; each → matching chunk(s)
    note: str = ""
    heading: str | None = None      # optional heading-path substring, disambiguates


@dataclass
class Resolved:
    question: GoldQuestion
    relevant: set[str] = field(default_factory=set)        # chunk_ids
    per_snippet: dict[str, list[int]] = field(default_factory=dict)  # snippet → chunk_indexes
    problems: list[str] = field(default_factory=list)


def _norm(text: str) -> str:
    """Whitespace-collapsed, case-folded — robust to markdown spacing noise."""
    return " ".join((text or "").split()).casefold()


def heading_path(payload: dict) -> str:
    parts = [payload.get(k) for k in ("h1_title", "h2_title", "h3_title")]
    return " > ".join(p for p in parts if p)


def load_goldset(path: Path = GOLDSET_PATH) -> list[GoldQuestion]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    out: list[GoldQuestion] = []
    seen: set[str] = set()
    for raw in data.get("q", []):
        q = GoldQuestion(
            id=raw["id"],
            file=raw["file"],
            question=raw["question"],
            gold=list(raw["gold"]),
            note=raw.get("note", ""),
            heading=raw.get("heading"),
        )
        if q.id in seen:
            raise ValueError(f"Duplicate golden-set id: {q.id!r}")
        seen.add(q.id)
        out.append(q)
    return out


async def fetch_chunks(client: AsyncQdrantClient | None = None) -> list[dict]:
    """Scroll every point's payload out of the collection."""
    client = client or get_qdrant_client()
    payloads: list[dict] = []
    offset = None
    while True:
        points, offset = await client.scroll(
            collection_name=settings.qdrant_collection,
            limit=256,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        payloads.extend(p.payload or {} for p in points)
        if offset is None:
            break
    return payloads


def resolve_one(q: GoldQuestion, chunks: list[dict]) -> Resolved:
    res = Resolved(question=q)
    scoped = [c for c in chunks if c.get("filename") == q.file]
    if not scoped:
        res.problems.append(f"no chunks for file {q.file!r}")
        return res
    nheading = _norm(q.heading) if q.heading else None
    for snippet in q.gold:
        ns = _norm(snippet)
        hits = [
            c for c in scoped
            if ns in _norm(c.get("content", ""))
            and (nheading is None or nheading in _norm(heading_path(c)))
        ]
        idxs = sorted(c.get("chunk_index", -1) for c in hits)
        res.per_snippet[snippet] = idxs
        if not hits:
            res.problems.append(f"snippet {snippet!r} matched 0 chunks")
        elif len(hits) > 3:
            res.problems.append(f"snippet {snippet!r} matched {len(hits)} chunks (too broad?)")
        for c in hits:
            res.relevant.add(c.get("chunk_id"))
    if not res.relevant:
        res.problems.append("no relevant chunks resolved")
    return res


def resolve_all(goldset: list[GoldQuestion], chunks: list[dict]) -> list[Resolved]:
    return [resolve_one(q, chunks) for q in goldset]


async def main() -> None:
    """Validation report: prints what each anchor resolved to."""
    goldset = load_goldset()
    chunks = await fetch_chunks()
    resolved = resolve_all(goldset, chunks)
    print(f"Golden set: {len(goldset)} questions · corpus: {len(chunks)} chunks\n")
    ok = 0
    for r in resolved:
        flag = "OK " if not r.problems else "!! "
        if not r.problems:
            ok += 1
        idxs = sorted({i for v in r.per_snippet.values() for i in v})
        print(f"[{flag}] {r.question.id:<22} → chunks {idxs}  ({len(r.relevant)} ids)")
        for p in r.problems:
            print(f"       ⚠ {p}")
    print(f"\n{ok}/{len(resolved)} questions resolved cleanly.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
