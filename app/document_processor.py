"""
document_processor.py
----------------------
Handles loading raw files (PDF, DOCX, TXT) and splitting them into
overlapping text chunks ready for embedding.

No external API calls here — everything runs locally and for free.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List
from pypdf import PdfReader
import docx


@dataclass
class Chunk:
    text: str
    source: str          # original filename
    chunk_id: int         # index within the document
    page: int | None = None


def load_pdf(path: Path) -> List[tuple[str, int]]:
    """Returns list of (page_text, page_number)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def load_docx(path: Path) -> List[tuple[str, int]]:
    document = docx.Document(str(path))
    full_text = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(full_text, None)]


def load_txt(path: Path) -> List[tuple[str, int]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(text, None)]


LOADERS = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".txt": load_txt,
}


def load_document(path: Path) -> List[tuple[str, int]]:
    ext = path.suffix.lower()
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}")
    return LOADERS[ext](path)


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[str]:
    """
    Simple recursive-ish splitter: tries to split on paragraph breaks first,
    falling back to raw character windows. This mirrors what LangChain's
    RecursiveCharacterTextSplitter does, without the dependency weight.
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " "]
    chunks: List[str] = []

    def split(segment: str, seps: List[str]) -> List[str]:
        if len(segment) <= chunk_size:
            return [segment] if segment.strip() else []
        if not seps:
            # hard cut with overlap
            result = []
            start = 0
            while start < len(segment):
                end = start + chunk_size
                result.append(segment[start:end])
                start = end - chunk_overlap
            return result

        sep, rest_seps = seps[0], seps[1:]
        parts = segment.split(sep)
        result = []
        buffer = ""
        for part in parts:
            candidate = (buffer + sep + part) if buffer else part
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                if buffer:
                    result.append(buffer)
                if len(part) > chunk_size:
                    result.extend(split(part, rest_seps))
                    buffer = ""
                else:
                    buffer = part
        if buffer:
            result.append(buffer)
        return result

    chunks = split(text, separators)
    return [c.strip() for c in chunks if c.strip()]


def process_file(path: Path, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Chunk]:
    """Load a file and return a list of Chunk objects ready for embedding."""
    pages = load_document(path)
    chunks: List[Chunk] = []
    cid = 0
    for page_text, page_num in pages:
        for piece in chunk_text(page_text, chunk_size, chunk_overlap):
            chunks.append(Chunk(text=piece, source=path.name, chunk_id=cid, page=page_num))
            cid += 1
    return chunks
