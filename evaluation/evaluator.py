import json
import time
from pathlib import Path
from typing import Dict, Any, List

from knowledge.rag import RAGEngine
from knowledge.vector_store import VectorStore
from synthesis.latex_compiler import LaTeXCompiler


class SystemEvaluator:
    """Evaluate the end-to-end system on a holdout set."""

    def __init__(self, store, rag: RAGEngine):
        self.store = store
        self.rag = rag

    def evaluate(self, holdout_topics: List[str],
                 k: int = 8) -> Dict[str, Any]:
        run_id = f"eval_{int(time.time())}"
        results = {
            "run_id": run_id,
            "topics": [],
            "summary": {},
        }

        precisions, recalls = [], []

        for topic in holdout_topics:
            hits = self.rag.retrieve(topic, top_k=k)
            topic_result = {
                "topic": topic,
                "retrieved": [
                    {"paper_id": h["metadata"].get("paper_id"),
                     "distance": h["distance"]}
                    for h in hits
                ],
            }

            # Precision@K: fraction of hits with distance < 0.5 (cosine)
            if hits:
                p_at_k = sum(1 for h in hits if h["distance"] < 0.5) / len(hits)
            else:
                p_at_k = 0.0
            precisions.append(p_at_k)
            topic_result["precision_at_k"] = p_at_k

            # Recall@K is only meaningful if we have ground-truth labels.
            # We proxy with "unique peer-reviewed sources found".
            unique_sources = {h["metadata"].get("paper_id") for h in hits}
            topic_result["unique_sources"] = len(unique_sources)
            recalls.append(min(1.0, len(unique_sources) / max(1, k)))

            results["topics"].append(topic_result)

        results["summary"] = {
            "mean_precision_at_k": (
                sum(precisions) / len(precisions) if precisions else 0.0
            ),
            "mean_recall_proxy": (
                sum(recalls) / len(recalls) if recalls else 0.0
            ),
            "topics_evaluated": len(holdout_topics),
            "k": k,
        }

        # LaTeX validation on the most recent paper
        paper = self.store.latest_paper()
        if paper:
            results["paper_check"] = {
                "experiment_id": paper.get("experiment_id"),
                "status": paper.get("status"),
                "latex_path": paper.get("latex_path"),
                "pdf_path": paper.get("pdf_path"),
                "pdf_exists": bool(paper.get("pdf_path")
                                   and Path(paper["pdf_path"]).exists()),
            }

        self.store.save_evaluation(run_id, results)
        return results

    def to_markdown(self, report: Dict[str, Any]) -> str:
        s = report.get("summary", {})
        lines = [
            f"# Evaluation Report ({report.get('run_id')})",
            "",
            f"- Topics evaluated: {s.get('topics_evaluated')}",
            f"- Mean Precision@{s.get('k')}: {s.get('mean_precision_at_k'):.3f}",
            f"- Mean Recall (proxy): {s.get('mean_recall_proxy'):.3f}",
        ]
        if "paper_check" in report:
            pc = report["paper_check"]
            lines += ["", "## Paper check",
                      f"- Experiment: {pc.get('experiment_id')}",
                      f"- Status: {pc.get('status')}",
                      f"- PDF exists: {pc.get('pdf_exists')}"]
        return "\n".join(lines)