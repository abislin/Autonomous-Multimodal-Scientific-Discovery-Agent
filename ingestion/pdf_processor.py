import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
import requests
from config import Config


class PDFProcessor:
    """Download PDFs and extract chunked text with metadata."""

    def download(self, pdf_url: str, paper_id: str) -> Path:
        Config.ensure_dirs()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", paper_id)
        out = Path(Config.PDF_DIR) / f"{safe_id}.pdf"
        if out.exists() and out.stat().st_size > 0:
            return out
        try:
            resp = requests.get(pdf_url, timeout=60)
            resp.raise_for_status()
            out.write_bytes(resp.content)
            return out
        except Exception as e:
            print(f"[PDF] download failed for {paper_id}: {e}")
            return None

    def extract_chunks(self, pdf_path: Path, meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        chunks = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[PDF] open failed {pdf_path}: {e}")
            return chunks

        buffer, buffer_pages = [], []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if not text.strip():
                continue
            buffer.append(text)
            buffer_pages.append(page_num)
            combined = "\n".join(buffer)
            while len(combined) >= Config.CHUNK_SIZE:
                chunk_text = combined[:Config.CHUNK_SIZE]
                self._emit(chunks, chunk_text, meta, buffer_pages[0])
                combined = combined[Config.CHUNK_SIZE - Config.CHUNK_OVERLAP:]
                buffer = [combined]
                buffer_pages = [page_num]
        if buffer:
            self._emit(chunks, "\n".join(buffer), meta, buffer_pages[0])
        doc.close()
        return chunks

    def _emit(self, chunks, text, meta, page):
        text = text.strip()
        if len(text) < 100:
            return
        chunk_id = hashlib.sha1(
            f"{meta['paper_id']}:{page}:{text[:80]}".encode()
        ).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": text,
            "metadata": {
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "authors": ", ".join(meta.get("authors", [])),
                "source": meta["source"],
                "published": meta.get("published", ""),
                "is_preprint": bool(meta.get("is_preprint", True)),
                "url": meta.get("entry_id", ""),
                "page": page,
            },
        })

    def abstract_chunk(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        """Always include abstract as a chunk even if PDF unavailable."""
        text = f"{meta['title']}\n\n{meta.get('abstract','')}"
        chunk_id = hashlib.sha1(f"{meta['paper_id']}:abstract".encode()).hexdigest()
        return {
            "id": chunk_id,
            "text": text,
            "metadata": {
                "paper_id": meta["paper_id"],
                "title": meta["title"],
                "authors": ", ".join(meta.get("authors", [])),
                "source": meta["source"],
                "published": meta.get("published", ""),
                "is_preprint": bool(meta.get("is_preprint", True)),
                "url": meta.get("entry_id", ""),
                "page": 0,
            },
        }