import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import Config
from knowledge.vector_store import VectorStore
from knowledge.rag import RAGEngine
from ingestion.pipeline import IngestionPipeline
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent
from agents.experiment_planner import ExperimentPlannerAgent
from agents.code_executor import CodeExecutorAgent
from agents.critic_agent import CriticAgent
from execution.docker_executor import DockerExecutor
from execution.sandbox import SandboxPolicy
from storage.experiment_store import ExperimentStore
from orchestration.workflow import DiscoveryWorkflow
from memory.research_memory import ResearchMemory
from synthesis.synthesis_agent import ResearchSynthesisAgent
from synthesis.latex_generator import LaTeXGenerator
from synthesis.latex_compiler import LaTeXCompiler
from evaluation.evaluator import SystemEvaluator


def build():
    Config.ensure_dirs()
    vs = VectorStore()
    rag = RAGEngine(vs)
    pipeline = IngestionPipeline(vs)
    store = ExperimentStore()
    reviewer = LiteratureReviewerAgent(rag)
    hypothesizer = HypothesisGeneratorAgent(rag)
    planner = ExperimentPlannerAgent()
    executor = CodeExecutorAgent(DockerExecutor(SandboxPolicy()), store)
    critic = CriticAgent()
    memory = ResearchMemory()
    synthesis = ResearchSynthesisAgent()
    return (pipeline, reviewer, hypothesizer, planner, executor,
            store, vs, rag, critic, memory, synthesis)


def cmd_discover(topic: str):
    (pipeline, reviewer, hyp, planner, executor,
     store, _, _, critic, memory, synth) = build()

    wf = DiscoveryWorkflow(
        pipeline, reviewer, hyp, planner, executor, store,
        critic=critic, memory=memory, synthesis=synth,
    )
    s = wf.run(topic)

    print("=== INGESTION ===")
    print(json.dumps(s.get("ingestion_stats", {}), indent=2))
    print("\n=== REVIEW (head) ===")
    print((s.get("review") or "")[:1200])
    print("\n=== HYPOTHESIS ===")
    print((s.get("chosen_hypothesis") or "")[:1000])
    print("\n=== PLAN ===")
    print(json.dumps(s.get("plan", {}), indent=2))
    print("\n=== EXECUTION ===")
    print(json.dumps({
        "ok": (s.get("code_result") or {}).get("ok"),
        "attempts": (s.get("code_result") or {}).get("attempts"),
        "metrics": (s.get("code_result") or {}).get("metrics"),
    }, indent=2))
    print("\n=== STATISTICS ===")
    print(json.dumps(s.get("stats_report", {}), indent=2))
    print("\n=== CRITIC VERDICT ===")
    print(json.dumps(s.get("critique", {}), indent=2))
    print("\n=== MEMORY ID ===")
    print(s.get("memory_id"))
    print("\n=== PAPER ===")
    print(json.dumps(s.get("paper", {}), indent=2))
    if s.get("error"):
        print("\nERROR:", s["error"])


def cmd_rerun(experiment_id: str):
    (_, _, _, _, executor, store, *_) = build()
    row = store.conn.execute(
        "SELECT plan_json FROM experiments WHERE id=?", (experiment_id,)
    ).fetchone()
    if not row:
        print("experiment not found")
        sys.exit(1)
    print(json.dumps(executor.run(experiment_id, json.loads(row[0])), indent=2))


def cmd_history(experiment_id: str):
    (_, _, _, _, _, store, *_) = build()
    print(json.dumps(store.history(experiment_id), indent=2))


def cmd_recall(question: str):
    mem = ResearchMemory()
    print(json.dumps(mem.recall(question, top_k=5), indent=2))


def cmd_synthesize(experiment_id: str):
    (_, _, _, _, _, store, _, rag, _, _, synth) = build()
    row = store.conn.execute(
        "SELECT topic, hypothesis, plan_json FROM experiments WHERE id=?",
        (experiment_id,),
    ).fetchone()
    if not row:
        print("experiment not found")
        sys.exit(1)
    topic, hypothesis, plan_json = row
    history = store.history(experiment_id)
    critique_row = store.conn.execute(
        "SELECT report_json FROM critiques WHERE experiment_id=? "
        "ORDER BY id DESC LIMIT 1", (experiment_id,),
    ).fetchone()
    stats_row = store.conn.execute(
        "SELECT report_json FROM stats_reports WHERE experiment_id=? "
        "ORDER BY id DESC LIMIT 1", (experiment_id,),
    ).fetchone()
    state = {
        "topic": topic,
        "chosen_hypothesis": hypothesis,
        "plan": json.loads(plan_json or "{}"),
        "code_result": {"metrics": (history[-1]["metrics"] if history else {})},
        "_history": history,
        "_rag_hits": rag.ret