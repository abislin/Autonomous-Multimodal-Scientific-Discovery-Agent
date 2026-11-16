import argparse
import json
import sys

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
    return (pipeline, reviewer, hypothesizer, planner, executor,
            store, vs, rag, critic, memory)


def cmd_discover(topic: str):
    (pipeline, reviewer, hyp, planner, executor,
     store, _, _, critic, memory) = build()

    wf = DiscoveryWorkflow(
        pipeline, reviewer, hyp, planner, executor, store,
        critic=critic, memory=memory,
    )
    s = wf.run(topic)

    print("=== INGESTION ===")
    print(json.dumps(s.get("ingestion_stats", {}), indent=2))

    print("\n=== REVIEW (head) ===")
    print((s.get("review") or "")[:1500])

    print("\n=== HYPOTHESIS ===")
    print((s.get("chosen_hypothesis") or "")[:1200])

    print("\n=== PLAN ===")
    print(json.dumps(s.get("plan", {}), indent=2))

    print("\n=== EXECUTION ===")
    print(json.dumps({
        "ok": (s.get("code_result") or {}).get("ok"),
        "attempts": (s.get("code_result") or {}).get("attempts"),
        "metrics": (s.get("code_result") or {}).get("metrics"),
        "run_id": (s.get("code_result") or {}).get("run_id"),
    }, indent=2))

    print("\n=== RL ===")
    print(json.dumps({
        "state": s.get("rl_state"),
        "action": s.get("rl_action"),
        "reward": s.get("rl_reward"),
    }, indent=2))

    print("\n=== FIGURES ===")
    print(json.dumps([
        {
            "figure_id": f.get("figure_id"),
            "chart_type": (f.get("vlm") or {}).get("chart_type"),
            "explanation": ((f.get("vlm") or {}).get("explanation") or "")[:300],
        }
        for f in s.get("figures", [])
    ], indent=2))

    print("\n=== STATISTICS ===")
    print(json.dumps(s.get("stats_report", {}), indent=2))

    print("\n=== CRITIC VERDICT ===")
    print(json.dumps(s.get("critique", {}), indent=2))

    print("\n=== MEMORY ID ===")
    print(s.get("memory_id"))

    if s.get("error"):
        print("\nERROR:", s["error"])


def cmd_rerun(experiment_id: str):
    """Reuse a stored plan and re-execute (useful for debugging)."""
    (_, _, _, _, executor, store, _, _, _, _) = build()
    row = store.conn.execute(
        "SELECT plan_json FROM experiments WHERE id=?", (experiment_id,)
    ).fetchone()
    if not row:
        print("experiment not found")
        sys.exit(1)
    plan = json.loads(row[0])
    result = executor.run(experiment_id=experiment_id, plan=plan)
    print(json.dumps(result, indent=2))


def cmd_history(experiment_id: str):
    (_, _, _, _, _, store, _, _, _, _) = build()
    print(json.dumps(store.history(experiment_id), indent=2))


def cmd_recall(question: str):
    mem = ResearchMemory()
    hits = mem.recall(question, top_k=5)
    print(json.dumps(hits, indent=2))


def main():
    p = argparse.ArgumentParser(
        description="Autonomous Multimodal Scientific Discovery Agent"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="Full pipeline: ingest → review → hypothesize → plan → execute → figures → stats → critique → memory")
    d.add_argument("--topic", required=True)

    r = sub.add_parser("rerun", help="Re-execute a stored experiment plan")
    r.add_argument("--experiment-id", required=True)

    h = sub.add_parser("history", help="Show run history for an experiment")
    h.add_argument("--experiment-id", required=True)

    q = sub.add_parser("recall", help="Search long-term research memory")
    q.add_argument("--question", required=True)

    args = p.parse_args()
    if args.cmd == "discover":
        cmd_discover(args.topic)
    elif args.cmd == "rerun":
        cmd_rerun(args.experiment_id)
    elif args.cmd == "history":
        cmd_history(args.experiment_id)
    elif args.cmd == "recall":
        cmd_recall(args.question)


if __name__ == "__main__":
    main()