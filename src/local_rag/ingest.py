from __future__ import annotations

from pathlib import Path

from local_rag.models import Chunk, Document

SUPPORTED = {".txt", ".md", ".pdf", ".pptx"}


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".pptx":
        return _extract_pptx(path)
    raise ValueError(f"formato no soportado: {suffix}")


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8")
        raise RuntimeError("pypdf no instalado y no hay sidecar .txt") from None
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError:
        sidecar = path.with_suffix(".txt")
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8")
        raise RuntimeError("python-pptx no instalado y no hay sidecar .txt") from None
    prs = Presentation(str(path))
    parts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                parts.append(shape.text)
    return "\n".join(parts)


def load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        if path.suffix.lower() in {".pdf", ".pptx"} and path.with_suffix(".txt").exists():
            continue
        docs.append(
            Document(
                doc_id=path.stem,
                path=str(path),
                text=extract_text(path),
            )
        )
    return docs


def chunk_document(doc: Document, size: int, overlap: int) -> list[Chunk]:
    text = " ".join(doc.text.split())
    if not text:
        return []
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(size - overlap, 1)
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{idx}",
                doc_id=doc.doc_id,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        if end == len(text):
            break
        start += step
        idx += 1
    return chunks


def ingest(docs_dir: Path, size: int = 280, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in load_documents(docs_dir):
        chunks.extend(chunk_document(doc, size=size, overlap=overlap))
    return chunks
