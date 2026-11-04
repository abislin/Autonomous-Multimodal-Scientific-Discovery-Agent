from datetime import datetime
from typing import List, Dict, Any
from Bio import Entrez
from config import Config

Entrez.email = "research-agent@example.com"


class PubMedClient:
    """Fetch papers from PubMed via Entrez API."""

    def search(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        max_results = max_results or Config.PUBMED_MAX_RESULTS
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        ids = record.get("IdList", [])
        if not ids:
            return []

        handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        papers = []
        for art in records.get("PubmedArticle", []):
            medline = art["MedlineCitation"]
            article = medline["Article"]
            pmid = str(medline["PMID"])
            title = str(article.get("ArticleTitle", "")).strip()
            abstract_parts = article.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts).strip()
            authors = []
            for a in article.get("AuthorList", []):
                last = a.get("LastName", "")
                init = a.get("Initials", "")
                if last:
                    authors.append(f"{last} {init}".strip())
            pub_date = article.get("Journal", {}).get("JournalIssue", {}) \
                              .get("PubDate", {})
            year = pub_date.get("Year", "")
            month = pub_date.get("Month", "")
            day = pub_date.get("Day", "")
            published = "-".join(x for x in [year, month, day] if x)

            papers.append({
                "paper_id": f"PMID:{pmid}",
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": published,
                "pdf_url": None,  # PubMed PDFs require subscription / PMC
                "entry_id": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "categories": [],
                "source": "PubMed",
                "is_preprint": False,  # PubMed indexes peer-reviewed literature
                "fetched_at": datetime.utcnow().isoformat(),
            })
        return papers