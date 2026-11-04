import uuid
from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from ingestion.pipeline import IngestionPipeline
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent
from agents.experiment_planner import ExperimentPlannerAgent
from agents.code_executor import CodeExecutorAgent
from storage.experiment_store import ExperimentStore
from rl.interface import RLState, RLAction, transition


class ResearchState(TypedDict, total=False):
    topic: str
    experiment_id: str
    ingestion_stats: Dict[str, int]
    review: str
    sources: list
    hypotheses: str
    chosen_hypothesis: str
    plan: Dict[str, Any]
    code_result: Dict[str, Any]
    rl_state: Dict[str, Any]
    rl_action: Dict[str, Any]
    rl_reward: float
    error: str


def _first_hypothesis_block(text: str) -> str:
    """Extract the first HYPOTHESIS: ... block for planning."""
    marker = "HYPOTHESIS:"
    if marker in text:
        start = text.index(marker)
        return text[start:start + 1200]
    return text[:1200]


class DiscoveryWorkflow:
    def __init__(self, pipeline: IngestionPipeline,
                 reviewer: LiteratureReviewerAgent,
                 hypothesizer: HypothesisGeneratorAgent,
                 planner: ExperimentPlannerAgent,
                 executor: CodeExecutorAgent,
                 store: ExperimentStore):
        self.pipeline, self.reviewer = pipeline, reviewer
        self.hypothesizer, self.planner = hypothesizer, planner
        self.executor, self.store = executor, store
        self.graph = self._build()

    def _ingest(self, s: ResearchState) -> ResearchState:
        try:
            s["ingestion_stats"] = self.pipeline.run(s["topic"])
        except Exception as e:
            s["error"] = f"ingest: {e}"
        return s

    def _review(self, s: ResearchState) -> ResearchState:
        if s.get("error"): return s
        out = self.reviewer.run(s["topic"])
        s["review"], s["sources"] = out["review"], out["sources"]
        return s

    def _hypothesize(self, s: ResearchState) -> ResearchState:
        if s.get("error"): return s
        out = self.hypothesizer.run(s["topic"])
        if out["hypotheses"].startswith("INSUFFICIENT_EVIDENCE"):
            s["error"] = "hypotheses: insufficient evidence"
            return s
        s["hypotheses"] = out["hypotheses"]
        s["sources"] = out.get("sources", s.get("sources", []))
        s["chosen_hypothesis"] = _first_hypothesis_block(out["hypotheses"])
        return s

    def _plan(self, s: ResearchState) -> ResearchState:
        if s.get("error"): return s
        plan = self.planner.run(s["chosen_hypothesis"])
        if plan.get("_parse_error"):
            s["error"] = "plan: JSON parse failed"
            return s
        s["plan"] = plan
        s["experiment_id"] = uuid.uuid4().hex[:12]
        self.store.create_experiment(
            s["experiment_id"], s["topic"], s["chosen_hypothesis"], plan
        )
        # RL observation snapshot (a real policy would pick the action here)
        s["rl_state"] = RLState(
            evidence_count=len(s.get("sources", [])),
            plan_task_type=plan.get("task_type", "classification"),
        ).__dict__
        s["rl_action"] = RLAction().asdict() if hasattr(RLAction(), "asdict") \
                          else RLAction().__dict__
        return s

    def _execute(self, s: ResearchState) -> ResearchState:
        if s.get("error"): return s
        result = self.executor.run(
            experiment_id=s["experiment_id"], plan=s["plan"]
        )
        s["code_result"] = result
        if not result["ok"]:
            s["error"] = f"execute: {result.get('error')}"
            return s
        reward = transition(
            state=RLState(**{k: v for k, v in s["rl_state"].items()
                             if k in RLState.__dataclass_fields__}),
            action=RLAction(),
            result=result,
            attempts=result["attempts"],
            duration_s=s["code_result"].get("duration_s", 0.0),
            budget_s=float(s["plan"].get("max_runtime_s", 180)),
        )
        s["rl_reward"] = reward.value()
        return s

    def _build(self):
        g = StateGraph(ResearchState)
        g.add_node("ingest", self._ingest)
        g.add_node("review", self._review)
        g.add_node("hypothesize", self._hypothesize)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.set_entry_point("ingest")
        g.add_edge("ingest", "review")
        g.add_edge("review", "hypothesize")
        g.add_edge("hypothesize", "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", END)
        return g.compile()

    def run(self, topic: str) -> ResearchState:
        return self.graph.invoke({"topic": topic})