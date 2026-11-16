import json
import re
from pathlib import Path
from typing import Dict, Any, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


TEMPLATES_DIR = Path(__file__).parent / "templates"


def _latex_escape(s: str) -> str:
    """Escape LaTeX special chars."""
    if s is None:
        return ""
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


def _safe_bibkey(paper_id: str) -> str:
    """Turn 'arXiv:2401.12345' or 'PMID:12345' into a bibtex-safe key."""
    key = re.sub(r"[^A-Za-z0-9]", "", paper_id or "ref")
    return f"ref{key}"


class LaTeXGenerator:
    def __init__(self, out_dir: Path = Path("artifacts/papers")):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(
            loader=FileSystemLoader(str(TEMPLATES_DIR)),
            autoescape=select_autoescape(disabled_extensions=("j2",)),
            block_start_string="<%",
            block_end_string="%>",
            variable_start_string="<<",
            variable_end_string=">>",
        )
        self.env.filters["latex"] = _latex_escape
        self.env.filters["bibkey"] = _safe_bibkey

    def build(self, experiment_id: str, state: Dict[str, Any],
              synthesis: Dict[str, Any]) -> Dict[str, str]:
        template = self.env.get_template("paper.tex.j2")

        refs = synthesis.get("references", []) or []
        bib_entries = []
        for r in refs:
            bib_entries.append({
                "key": _safe_bibkey(r.get("paper_id", "ref")),
                "paper_id": r.get("paper_id", "?"),
                "title": r.get("title", "Untitled"),
                "authors": r.get("authors", "Unknown"),
                "year": r.get("year", ""),
                "venue": r.get("venue", r.get("source", "")),
                "url": r.get("url", ""),
            })

        figures = []
        for i, f in enumerate(state.get("figures", []) or []):
            img = f.get("image_path", "")
            if not img or not Path(img).exists():
                continue
            rel = Path(img).name
            figures.append({
                "label": f"fig:auto{i+1}",
                "path": rel,
                "caption": (f.get("caption")
                            or (f.get("vlm") or {}).get("explanation", "")[:200]),
                "chart_type": (f.get("vlm") or {}).get("chart_type", "figure"),
            })
        fig_dir = self.out_dir / f"{experiment_id}_figs"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(figures):
            src = Path(state["figures"][i]["image_path"])
            if src.exists():
                (fig_dir / f["path"]).write_bytes(src.read_bytes())

        metrics = (state.get("code_result") or {}).get("metrics", {}) or {}
        results_rows = [
            {"metric": k, "value": f"{v:.4f}" if isinstance(v, (int, float)) else str(v)}
            for k, v in metrics.items()
        ]

        history = state.get("_history", []) or []
        history_rows = [
            {"attempt": h.get("attempt"), "ok": h.get("ok"),
             "duration": h.get("duration_s")}
            for h in history
        ]

        stats = state.get("stats_report", {}) or {}
        critique = state.get("critique", {}) or {}

        rendered = template.render(
            title=synthesis.get("title", state.get("topic", "Untitled")),
            abstract=synthesis.get("abstract", ""),
            introduction=synthesis.get("introduction", ""),
            related_work=synthesis.get("related_work", ""),
            methodology=synthesis.get("methodology", ""),
            experimental_setup=synthesis.get("experimental_setup", ""),
            results=synthesis.get("results", ""),
            statistical_analysis=synthesis.get("statistical_analysis", ""),
            discussion=synthesis.get("discussion", ""),
            conclusion=synthesis.get("conclusion", ""),
            hypothesis=state.get("chosen_hypothesis", ""),
            plan=state.get("plan", {}),
            metrics=results_rows,
            history=history_rows,
            stats=stats,
            critique=critique,
            figures=figures,
            references=bib_entries,
            fig_dir=fig_dir.name,
            topic=state.get("topic", ""),
            experiment_id=experiment_id,
        )

        tex_path = self.out_dir / f"{experiment_id}.tex"
        bib_path = self.out_dir / f"{experiment_id}.bib"
        tex_path.write_text(rendered, encoding="utf-8")
        bib_path.write_text(self._render_bib(bib_entries), encoding="utf-8")

        return {
            "tex_path": str(tex_path),
            "bib_path": str(bib_path),
            "fig_dir": str(fig_dir),
            "title": synthesis.get("title", ""),
            "abstract": synthesis.get("abstract", ""),
        }

    @staticmethod
    def _render_bib(entries: List[Dict[str, Any]]) -> str:
        chunks = []
        for e in entries:
            authors = e.get("authors") or "Unknown"
            if isinstance(authors, list):
                authors = " and ".join(authors)
            chunks.append(
                f"@misc{{{e['key']},\n"
                f"  title  = {{{e['title']}}},\n"
                f"  author = {{{authors}}},\n"
                f"  year   = {{{e['year']}}},\n"
                f"  note   = {{{e['paper_id']}}},\n"
                f"  url    = {{{e['url']}}}\n"
                f"}}\n"
            )
        return "\n".join(chunks)