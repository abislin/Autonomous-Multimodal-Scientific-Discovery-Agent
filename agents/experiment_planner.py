import json
import re
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

SYSTEM = """You are an Experiment Planner Agent.
Convert one hypothesis into a STRICT JSON experiment plan with these keys:
{
  "hypothesis_id": str,
  "task_type": "classification"|"regression",
  "dataset": {"name": str, "source": "sklearn"|"synthetic", "notes": str},
  "baseline_model": str,
  "proposed_model": str,
  "preprocessing": [str, ...],
  "evaluation_metrics": [str, ...],   # Accuracy, Precision, Recall, F1, MAE, RMSE
  "train_test_split": {"test_size": float, "random_state": int},
  "success_criterion": {"metric": str, "threshold": float},
  "max_runtime_s": int,
  "allowed_libraries": ["numpy","pandas","sklearn","scipy"]
}
Return ONLY valid JSON. No prose. No markdown.
"""

USER = """Hypothesis:
{hypothesis}

Produce the JSON experiment plan."""


class ExperimentPlannerAgent(BaseAgent):
    name = "experiment_planner"
    system_prompt = SYSTEM

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM), ("human", USER),
        ])

    def run(self, hypothesis: str) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        resp = chain.invoke({"hypothesis": hypothesis})
        raw = resp.content.strip()
        # Strip accidental code fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            plan = {"_parse_error": True, "raw": raw}
        return plan