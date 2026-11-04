import arxiv
from datetime import datetime
from typing import List, Dict, Any
from config import Config


class ArxivClient:
    """Fetch papers from arXiv API."""

    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        max_results = max_results or Config.ARXIV_MAX_RESULTS
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for r in client.results(search):
            papers.append({
                "paper_id": r.get_short_id(),
                "title": r.title.strip(),
                "abstract": r.summary.strip(),
                "authors": [a.name for a in r.authors],
                "published": r.published.strftime("%Y-%m-%d") if r.published else "",
                "updated": r.updated.strftime("%Y-%m-%d") if r.updated else "",
                "pdf_url": r.pdf_url,
                "entry_id": r.entry_id,
                "categories": r.categories,
                "source": "arXiv",
                "is_preprint": True,  # arXiv is a preprint server
                "fetched_at": datetime.utcnow().isoformat(),
            })
        return papers