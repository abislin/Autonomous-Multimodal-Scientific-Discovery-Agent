import argparse
import json
import sys
from config import Config
from knowledge.vector_store import VectorStore
from knowledge.rag import RAGEngine
from ingestion.pipeline import IngestionPipeline
from agents.literature_reviewer import LiteratureReviewerAgent
from agents.hypothesis_generator import HypothesisGeneratorAgent
from orchestration.workflow import DiscoveryWorkflow


def build_system():
    Config.ensure_dirs()
    vs = VectorStore()
    rag = RAGEngine(vs)
    pipeline = IngestionPipeline(vs)
    reviewer = LiteratureReviewerAgent(rag)
    hypothesizer = HypothesisGeneratorAgent()
    return vs, pipeline, reviewer, hypothesizer, rag


def cmd_ingest(topic: str):
    vs, pipeline, *_ = build_system()
    stats = pipeline.run(topic)
    print(json.dumps(stats, indent=2))


def cmd_query(question: str):
    vs, _, _, _, rag = build_system()
    if vs.count() == 0:
        print("Knowledge base is empty. Run `ingest` first.")
        sys.exit(1)
    out = rag.answer(question)
    print("=== ANSWER ===\n" + out["answer"])
    print("\n=== SOURCES ===\n" + ", ".join(out["sources"]))


def cmd_discover(topic: str):
    vs, pipeline, reviewer, hypothesizer, _ = build_system()
    wf = DiscoveryWorkflow(pipeline, reviewer, hypothesizer)
    state = wf.run(topic)
    if state.get("error"):
        print("ERROR:", state["error"])
        sys.exit(1)
    print("=== INGESTION ===")
    print(json.dumps(state["ingestion_stats"], indent=2))
    print("\n=== LITERATURE REVIEW ===")
    print(state["review"])
    print("\n=== SOURCES ===")
    print(", ".join(state.get("sources", [])))
    print("\n=== HYPOTHESES ===")
    print(state["hypotheses"])


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Multimodal Scientific Discovery Agent (Week 1)"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("ingest", help="Fetch papers and populate the vector DB")
    p1.add_argument("--topic", required=True)

    p2 = sub.add_parser("query", help="RAG query over the knowledge base")
    p2.add_argument("--question", required=True)

    p3 = sub.add_parser("discover", help="Full pipeline: ingest → review → hypothesize")
    p3.add_argument("--topic", required=True)

    args = parser.parse_args()
    if args.cmd == "ingest":
        cmd_ingest(args.topic)
    elif args.cmd == "query":
        cmd_query(args.question)
    elif args.cmd == "discover":
        cmd_discover(args.topic)


if __name__ == "__main__":
    main()