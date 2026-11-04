from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from ingestion.pipeline import IngestionPipeline
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent


class ResearchState(TypedDict, total=False):
    topic: str
    ingestion_stats: Dict[str, int]
    review: str
    sources: list
    hypotheses: str
    error: str


class DiscoveryWorkflow:
    """LangGraph orchestration: ingest → review → hypothesize."""

    def __init__(self, pipeline: IngestionPipeline,
                 reviewer: LiteratureReviewerAgent,
                 hypothesizer: HypothesisGeneratorAgent):
        self.pipeline = pipeline
        self.reviewer = reviewer
        self.hypothesizer = hypothesizer
        self.graph = self._build()

    def _ingest(self, state: ResearchState) -> ResearchState:
        try:
            stats = self.pipeline.run(state["topic"])
            state["ingestion_stats"] = stats
        except Exception as e:
            state["error"] = f"ingestion failed: {e}"
        return state

    def _review(self, state: ResearchState) -> ResearchState:
        if state.get("error"):
            return state
        out = self.reviewer.run(state["topic"])
        state["review"] = out["review"]
        state["sources"] = out["sources"]
        return state

    def _hypothesize(self, state: ResearchState) -> ResearchState:
        if state.get("error") or not state.get("review"):
            return state
        out = self.hypothesizer.run(
            state["topic"], state["review"], state.get("sources", [])
        )
        state["hypotheses"] = out["hypotheses"]
        return state

    def _build(self):
        g = StateGraph(ResearchState)
        g.add_node("ingest", self._ingest)
        g.add_node("review", self._review)
        g.add_node("hypothesize", self._hypothesize)
        g.set_entry_point("ingest")
        g.add_edge("ingest", "review")
        g.add_edge("review", "hypothesize")
        g.add_edge("hypothesize", END)
        return g.compile()

    def run(self, topic: str) -> ResearchState:
        return self.graph.invoke({"topic": topic})