import json
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent


SYSTEM = """You are a Research Synthesis Agent for a scientific ML project.
You are given:
  (a) the research topic,
  (b) the literature review (with [paper_id] citations),
  (c) the chosen hypothesis,
  (d) the experiment plan,
  (e) the execution metrics and run history,
  (f) statistical validation output,
  (g) the Critic Agent report,
  (h) the figure summaries produced by a VLM.

Produce a structured, honest research paper draft. Ground every factual
claim in the provided evidence; never invent numbers or citations.
Explicitly state limitations and do NOT claim independent scientific
validation.

Return STRICT JSON with keys:
{
  "title": str,
  "abstract": str,
  "introduction": str,
  "related_work": str,
  "methodology": str,
  "experimental_setup": str,
  "results": str,
  "statistical_analysis": str,
  "discussion": str,
  "conclusion": str,
  "references": [{"paper_id":str,"title":str,"authors":str,"year":str,
                  "venue":str,"url":str}, ...]
}
Return ONLY the JSON object, no markdown fences."""

USER = """Topic: {topic}

Literature review:
{review}

Chosen hypothesis:
{hypothesis}

Experiment plan:
{plan}

Execution metrics:
{metrics}

Run history:
{history}

Statistical validation:
{stats}

Critic report:
{critique}

Figures (VLM summaries):
{figures}

Write the paper draft JSON."""


class ResearchSynthesisAgent(BaseAgent):
    name = "research_synthesis"
    system_prompt = SYSTEM

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", USER),
        ])

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        source_meta = {}
        for h in state.get("_rag_hits", []) or []:
            m = h.get("metadata", {})
            pid = m.get("paper_id")
            if pid and pid not in source_meta:
                source_meta[pid] = {
                    "paper_id": pid,
                    "title": m.get("title", ""),
                    "authors": m.get("authors", ""),
                    "year": (m.get("published", "") or "")[:4],
                    "venue": m.get("source", ""),
                    "url": m.get("url", ""),
                }

        figures_summary = [
            {
                "figure_id": f.get("figure_id"),
                "caption": f.get("caption") or "",
                "chart_type": (f.get("vlm") or {}).get("chart_type"),
                "explanation": (f.get("vlm") or {}).get("explanation", ""),
            }
            for f in (state.get("figures") or [])
        ]

        chain = self.prompt | self.llm
        resp = chain.invoke({
            "topic": state.get("topic", ""),
            "review": (state.get("review") or "")[:6000],
            "hypothesis": (state.get("chosen_hypothesis") or "")[:2000],
            "plan": json.dumps(state.get("plan", {}), indent=2)[:3000],
            "metrics": json.dumps(
                (state.get("code_result") or {}).get("metrics", {}), indent=2
            ),
            "history": json.dumps(state.get("_history", []), indent=2)[:4000],
            "stats": json.dumps(state.get("stats_report", {}), indent=2)[:3000],
            "critique": json.dumps(state.get("critique", {}), indent=2)[:3000],
            "figures": json.dumps(figures_summary, indent=2)[:3000],
        })

        raw = (resp.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "_parse_error": True,
                "raw": raw[:4000],
                "title": state.get("topic", "Untitled"),
                "abstract": "Synthesis output could not be parsed as JSON.",
                "references": list(source_meta.values()),
            }

        merged_refs = []
        seen = set()
        for r in data.get("references", []) or []:
            pid = r.get("paper_id")
            if pid and pid in source_meta:
                merged_refs.append(source_meta[pid])
                seen.add(pid)
        for pid, meta in source_meta.items():
            if pid not in seen:
                merged_refs.append(meta)
        data["references"] = merged_refs
        return data