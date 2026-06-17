from pathlib import Path

from .chunker import Chunk, chunk_markdown


def load_markdown(file_path: str | Path) -> list[Chunk]:
    text = Path(file_path).read_text(encoding="utf-8")
    return chunk_markdown(text)


def load_markdown_text(text: str) -> list[Chunk]:
    return chunk_markdown(text)
