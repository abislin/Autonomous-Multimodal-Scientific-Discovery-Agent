from typing import List, Dict, Any
from ingestion.arxiv_client import ArxivClient
from ingestion.pubmed_client import PubMedClient
from ingestion.pdf_processor import PDFProcessor
from knowledge.vector_store import VectorStore


class IngestionPipeline:
    """Collect, dedupe, process and store scientific literature."""

    def __init__(self, vector_store: VectorStore):
        self.arxiv = ArxivClient()
        self.pubmed = PubMedClient()
        self.pdf = PDFProcessor()
        self.vs = vector_store
        self._seen_ids = set()

    def run(self, topic: str, use_arxiv: bool = True, use_pubmed: bool = True
            ) -> Dict[str, int]:
        papers: List[Dict[str, Any]] = []
        if use_arxiv:
            try:
                papers += self.arxiv.search(topic)
            except Exception as e:
                print(f"[arXiv] error: {e}")
        if use_pubmed:
            try:
                papers += self.pubmed.search(topic)
            except Exception as e:
                print(f"[PubMed] error: {e}")

        # Deduplicate by paper_id
        unique = []
        for p in papers:
            if p["paper_id"] in self._seen_ids:
                continue
            self._seen_ids.add(p["paper_id"])
            unique.append(p)

        stored_chunks = 0
        for p in unique:
            chunks = [self.pdf.abstract_chunk(p)]
            if p.get("pdf_url"):
                path = self.pdf.download(p["pdf_url"], p["paper_id"])
                if path:
                    chunks += self.pdf.extract_chunks(path, p)
            self.vs.add_chunks(chunks)
            stored_chunks += len(chunks)

        return {
            "papers_fetched": len(papers),
            "papers_unique": len(unique),
            "chunks_stored": stored_chunks,
            "total_in_db": self.vs.count(),
        }