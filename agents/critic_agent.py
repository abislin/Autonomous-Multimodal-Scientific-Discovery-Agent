import json
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent


SYSTEM = """You are a Critic Agent for scientific machine-learning experiments.
You are given: (a) the hypothesis, (b) the experiment plan, (c) the run history
(codes, metrics, errors), and (d) the statistical validation report.

Your job is to be SKEPTICAL and rigorous. Investigate specifically:
1. DATA LEAKAGE: was feature selection / preprocessing fit on the test set?
2. TRAIN/TEST SPLIT: same split for baseline and proposed model?
3. METRIC APPROPRIATENESS: e.g. accuracy on imbalanced data is misleading.
4. STATISTICAL SIGNIFICANCE: does the improvement survive the reported p-value / CI?
5. CONCLUSION VALIDITY: is "Model A beats Model B" actually supported?
6. FAILURE ANALYSIS: were any runs discarded after seeing results (cherry-picking)?
7. GENERALIZATION RISK: single split, no cross-validation, tiny dataset, etc.

Return STRICT JSON:
{
  "verdict": "supported" | "weakly_supported" | "not_supported" | "inconclusive",
  "severity": "low" | "medium" | "high",
  "concerns": [{"issue": str, "evidence": str, "suggestion": str}, ...],
  "strengths": [str, ...],
  "methodology_score": float,     // 0.0 - 1.0
  "significance_assessment": str,
  "final_report": str             // 4-6 sentence human-readable summary
}
Return ONLY the JSON object."""

USER = """Hypothesis:
{hypothesis}

Experiment plan:
{plan}

Run history (all attempts, in order):
{history}

Statistical validation:
{stats}

Produce the JSON critique."""


class CriticAgent(BaseAgent):
    name = "critic_agent"
    system_prompt = SYSTEM

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", USER),
        ])

    def run(self, hypothesis: str, plan: Dict[str, Any],
            history: List[Dict[str, Any]],
            stats: Dict[str, Any]) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        resp = chain.invoke({
            "hypothesis": hypothesis,
            "plan": json.dumps(plan, indent=2)[:4000],
            "history": json.dumps(history, indent=2)[:6000],
            "stats": json.dumps(stats, indent=2)[:4000],
        })
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            if raw.rstrip().endswith("```"):
                raw = raw.rstrip()[:-3]
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "_parse_error": True,
                "raw": raw[:3000],
                "verdict": "inconclusive",
                "severity": "medium",
                "concerns": [],
                "final_report": "Critic output could not be parsed as JSON.",
            }