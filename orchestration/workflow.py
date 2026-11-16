import json
import uuid
from pathlib import Path
from typing import Dict, Any, TypedDict

from langgraph.graph import StateGraph, END

from config import Config
from ingestion.pipeline import IngestionPipeline
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent
from agents.experiment_planner import ExperimentPlannerAgent
from agents.code_executor import CodeExecutorAgent
from agents.critic_agent import CriticAgent
from storage.experiment_store import ExperimentStore
from rl.interface import RLState, RLAction, transition
from multimodal.figure_extractor import FigureExtractor
from multimodal.chart_classifier import ChartClassifier
from multimodal.vlm_client import VLMClient
from statistics.validation import StatisticalValidator
from memory.research_memory import ResearchMemory
from synthesis.synthesis_agent import ResearchSynthesisAgent
from synthesis.latex_generator import LaTeXGenerator
from synthesis.latex_compiler import LaTeXCompiler


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
    figures: list
    critique: Dict[str, Any]
    stats_report: Dict[str, Any]
    memory_id: str
    _history: list
    _rag_hits: list
    synthesis: Dict[str, Any]
    paper: Dict[str, Any]
    tex_path: str
    pdf_path: str
    error: str


def _first_hypothesis_block(text: str) -> str:
    marker = "HYPOTHESIS:"
    if marker in text:
        start = text.index(marker)
        return text[start:start + 1200]
    return text[:1200]


class DiscoveryWorkflow:
    def __init__(self,
                 pipeline: IngestionPipeline,
                 reviewer: LiteratureReviewerAgent,
                 hypothesizer: HypothesisGeneratorAgent,
                 planner: ExperimentPlannerAgent,
                 executor: CodeExecutorAgent,
                 store: ExperimentStore,
                 critic: CriticAgent = None,
                 memory: ResearchMemory = None,
                 synthesis: ResearchSynthesisAgent = None):
        self.pipeline = pipeline
        self.reviewer = reviewer
        self.hypothesizer = hypothesizer
        self.planner = planner
        self.executor = executor
        self.store = store
        self.critic = critic or CriticAgent()
        self.memory = memory or ResearchMemory()
        self.synthesis = synthesis or ResearchSynthesisAgent()
        self.graph = self._build()

    # ---------------- Week 1/2 nodes ----------------
    def _ingest(self, s: ResearchState) -> ResearchState:
        try:
            s["ingestion_stats"] = self.pipeline.run(s["topic"])
        except Exception as e:
            s["error"] = f"ingest: {e}"
        return s

    def _review(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        out = self.reviewer.run(s["topic"])
        s["review"] = out["review"]
        s["sources"] = out["sources"]
        try:
            s["_rag_hits"] = self.reviewer.rag.retrieve(s["topic"], top_k=10)
        except Exception:
            s["_rag_hits"] = []
        return s

    def _hypothesize(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        out = self.hypothesizer.run(s["topic"])
        if (out.get("hypotheses") or "").startswith("INSUFFICIENT_EVIDENCE"):
            s["error"] = "hypotheses: insufficient evidence"
            return s
        s["hypotheses"] = out["hypotheses"]
        s["sources"] = out.get("sources", s.get("sources", []))
        s["chosen_hypothesis"] = _first_hypothesis_block(out["hypotheses"])
        return s

    def _plan(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        plan = self.planner.run(s["chosen_hypothesis"])
        if plan.get("_parse_error"):
            s["error"] = "plan: JSON parse failed"
            return s
        s["plan"] = plan
        s["experiment_id"] = uuid.uuid4().hex[:12]
        self.store.create_experiment(
            s["experiment_id"], s["topic"], s["chosen_hypothesis"], plan
        )
        rl_state = RLState(
            evidence_count=len(s.get("sources", [])),
            plan_task_type=plan.get("task_type", "classification"),
        )
        s["rl_state"] = rl_state.__dict__
        s["rl_action"] = RLAction().__dict__
        return s

    def _execute(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        result = self.executor.run(
            experiment_id=s["experiment_id"], plan=s["plan"]
        )
        s["code_result"] = result
        if not result.get("ok"):
            s["error"] = f"execute: {result.get('error')}"
            return s
        try:
            rl_state = RLState(**{
                k: v for k, v in s["rl_state"].items()
                if k in RLState.__dataclass_fields__
            })
            reward = transition(
                state=rl_state,
                action=RLAction(),
                result=result,
                attempts=result.get("attempts", 1),
                duration_s=result.get("duration_s", 0.0),
                budget_s=float(s["plan"].get("max_runtime_s", 180)),
            )
            s["rl_reward"] = reward.value()
        except Exception:
            s["rl_reward"] = 0.0
        return s

    # ---------------- Week 3 nodes ----------------
    def _analyse_figures(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        try:
            figs = []
            extractor = FigureExtractor()
            classifier = ChartClassifier()
            vlm = VLMClient()
            for pdf in Path(Config.PDF_DIR).glob("*.pdf"):
                raw = extractor.extract(pdf, paper_id=pdf.stem)
                for f in raw:
                    hint = classifier.classify(f["image_path"])
                    vlm_out = vlm.describe(
                        f["image_path"], f.get("caption") or ""
                    )
                    vlm_out.setdefault("chart_type", hint["chart_type"])
                    vlm_out["heuristic_chart_type"] = hint["chart_type"]
                    self.store.save_figure(
                        s["experiment_id"], f, vlm_out
                    )
                    f["vlm"] = vlm_out
                    figs.append(f)
            s["figures"] = figs
        except Exception as e:
            s["error"] = f"figures: {e}"
        return s

    def _validate_stats(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        try:
            history = self.store.history(s["experiment_id"])
            ok_runs = [h for h in history if h.get("ok")]
            scores = [
                h["metrics"].get(
                    "accuracy",
                    h["metrics"].get("f1", h["metrics"].get("r2", 0.0)),
                )
                for h in ok_runs
            ]
            if len(scores) < 2:
                s["stats_report"] = {
                    "comparison": {
                        "test": "none",
                        "p_value": 1.0,
                        "significant": False,
                    },
                    "interpretation":
                        "Need >=2 successful runs for any comparison.",
                }
            else:
                mid = len(scores) // 2
                baseline = scores[:mid] or scores[:1]
                proposed = scores[mid:] or scores[-1:]
                validator = StatisticalValidator()
                s["stats_report"] = validator.report(
                    baseline, proposed, design_hint=None
                )
            self.store.save_stats(
                s["experiment_id"], s["stats_report"]
            )
        except Exception as e:
            s["error"] = f"stats: {e}"
        return s

    def _critique(self, s: ResearchState) -> ResearchState:
        err = s.get("error") or ""
        if err and "execute" not in err:
            return s
        try:
            history = self.store.history(s["experiment_id"])
            report = self.critic.run(
                hypothesis=s.get("chosen_hypothesis", ""),
                plan=s.get("plan", {}),
                history=history,
                stats=s.get("stats_report", {}),
            )
            s["critique"] = report
            self.store.save_critique(s["experiment_id"], report)
        except Exception as e:
            s["error"] = f"critique: {e}"
        return s

    def _remember(self, s: ResearchState) -> ResearchState:
        try:
            existing = self.memory.has_similar_experiment(s["topic"])
            if existing:
                s["memory_id"] = existing["id"]
                return s
            blob = json.dumps({
                "topic": s["topic"],
                "hypothesis": (s.get("chosen_hypothesis") or "")[:800],
                "metrics": (s.get("code_result") or {}).get("metrics", {}),
                "verdict": (s.get("critique") or {}).get("verdict"),
                "p_value": (s.get("stats_report") or {})
                    .get("comparison", {}).get("p_value"),
            })
            s["memory_id"] = self.memory.remember(
                kind="result", topic=s["topic"], text=blob,
                metadata={"experiment_id": s["experiment_id"]},
            )
        except Exception as e:
            s["error"] = f"memory: {e}"
        return s

    # ---------------- Week 4 nodes ----------------
    def _collect_context(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        try:
            s["_history"] = self.store.history(s.get("experiment_id", ""))
            if "_rag_hits" not in s or not s["_rag_hits"]:
                try:
                    s["_rag_hits"] = self.reviewer.rag.retrieve(
                        s["topic"], top_k=10
                    )
                except Exception:
                    s["_rag_hits"] = []
        except Exception as e:
            s["error"] = f"collect: {e}"
        return s

    def _synthesize(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        try:
            out = self.synthesis.run(s)
            s["synthesis"] = out
            self.store.save_synthesis(s.get("experiment_id", ""), out)
        except Exception as e:
            s["error"] = f"synthesis: {e}"
        return s

    def _generate_latex(self, s: ResearchState) -> ResearchState:
        if s.get("error"):
            return s
        try:
            gen = LaTeXGenerator()
            built = gen.build(
                experiment_id=s.get("experiment_id", "exp"),
                state=s,
                synthesis=s.get("synthesis", {}),
            )
            s["tex_path"] = built["tex_path"]
            s["paper"] = {
                "tex_path": built["tex_path"],
                "bib_path": built["bib_path"],
                "title": built["title"],
                "abstract": built["abstract"],
            }
            compiler = LaTeXCompiler()
            compiled = compiler.compile(built["tex_path"])
            s["paper"]["pdf_path"] = compiled.get("pdf_path")
            s["paper"]["compile"] = {
                "ok": compiled.get("ok"),
                "errors": compiled.get("errors", []),
                "warnings": compiled.get("warnings", []),
            }
            s["pdf_path"] = compiled.get("pdf_path")
            self.store.save_paper(
                experiment_id=s.get("experiment_id", "exp"),
                topic=s.get("topic", ""),
                latex_path=built["tex_path"],
                pdf_path=compiled.get("pdf_path") or "",
                abstract=built["abstract"] or "",
                status="compiled" if compiled.get("ok") else "generated",
            )
        except Exception as e:
            s["error"] = f"latex: {e}"
        return s

    # ---------------- graph ----------------
    def _build(self):
        g = StateGraph(ResearchState)
        g.add_node("ingest", self._ingest)
        g.add_node("review", self._review)
        g.add_node("hypothesize", self._hypothesize)
        g.add_node("plan", self._plan)
        g.add_node("execute", self._execute)
        g.add_node("figures", self._analyse_figures)
        g.add_node("stats", self._validate_stats)
        g.add_node("critique", self._critique)
        g.add_node("remember", self._remember)
        g.add_node("collect", self._collect_context)
        g.add_node("synthesize", self._synthesize)
        g.add_node("latex", self._generate_latex)

        g.set_entry_point("ingest")
        g.add_edge("ingest", "review")
        g.add_edge("review", "hypothesize")
        g.add_edge("hypothesize", "plan")
        g.add_edge("plan", "execute")
        g.add_edge("execute", "figures")
        g.add_edge("figures", "stats")
        g.add_edge("stats", "critique")
        g.add_edge("critique", "remember")
        g.add_edge("remember", "collect")
        g.add_edge("collect", "synthesize")
        g.add_edge("synthesize", "latex")
        g.add_edge("latex", END)
        return g.compile()

    def run(self, topic: str) -> ResearchState:
        return self.graph.invoke({"topic": topic})