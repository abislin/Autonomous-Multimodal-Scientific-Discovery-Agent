from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent
from knowledge.rag import RAGEngine

SYSTEM = """You are a Hypothesis Generator Agent grounded STRICTLY in retrieved
peer-reviewed evidence. Rules:
1. Every hypothesis MUST be supported by at least one [paper_id] from the evidence.
2. Prefer PEER-REVIEWED over PREPRINT sources; label both explicitly.
3. If evidence is insufficient, respond "INSUFFICIENT_EVIDENCE" and stop.
4. For each hypothesis output exactly these fields:
   - HYPOTHESIS:
   - RATIONALE (must cite [paper_id]):
   - GAP ADDRESSED:
   - PROPOSED TEST:
   - EXPECTED METRIC(S):
   - SUPPORTING_REFS: [paper_id, ...]
"""

USER = """Topic: {topic}

Retrieved evidence:
{context}

Produce 3 testable hypotheses or INSUFFICIENT_EVIDENCE."""


class HypothesisGeneratorAgent(BaseAgent):
    name = "hypothesis_generator"
    system_prompt = SYSTEM

    def __init__(self, rag: RAGEngine):
        super().__init__()
        self.rag = rag
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM), ("human", USER),
        ])

    def _format(self, hits: List[Dict[str, Any]]) -> str:
        blocks = []
        for h in hits:
            m = h["metadata"]
            tag = "PREPRINT" if m.get("is_preprint") else "PEER-REVIEWED"
            blocks.append(
                f"[{m.get('paper_id')}] ({tag}) {m.get('title')} "
                f"({m.get('source')}, {m.get('published')})\n{h['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def run(self, topic: str, top_k: int = 10) -> Dict[str, Any]:
        hits = self.rag.retrieve(topic, top_k=top_k)
        if not hits:
            return {"hypotheses": "INSUFFICIENT_EVIDENCE", "sources": [], "hits": []}
        context = self._format(hits)
        chain = self.prompt | self.llm
        resp = chain.invoke({"topic": topic, "context": context})
        sources = sorted({h["metadata"].get("paper_id") for h in hits})
        return {
            "hypotheses": resp.content,
            "sources": sources,
            "hits": hits,
            "evidence_count": len(hits),
        }