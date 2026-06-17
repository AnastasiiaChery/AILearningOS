from pathlib import Path

import pdfplumber

from .chunker import Chunk, chunk_pdf_pages


def load_pdf(file_path: str | Path) -> list[Chunk]:
    pages: list[tuple[int, str]] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Also extract tables as plain text
            tables = page.extract_tables()
            table_texts = []
            for table in tables:
                rows = [" | ".join(str(cell or "") for cell in row) for row in table if row]
                table_texts.append("\n".join(rows))
            combined = text + ("\n\n" + "\n\n".join(table_texts) if table_texts else "")
            pages.append((i, combined.strip()))
    return chunk_pdf_pages(pages)
