from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent
from knowledge.rag import RAGEngine

SYSTEM = """You are a Literature Reviewer Agent.
Analyze the retrieved scientific evidence and produce a structured review.
Rules:
- Cite every claim with [paper_id].
- Distinguish PEER-REVIEWED vs PREPRINT evidence.
- Be objective; note contradictions between studies.
Return these sections:
1. Overview
2. Key Findings
3. Methodologies
4. Limitations & Contradictions
5. Research Gaps
6. Source List
"""

USER = """Research topic: {topic}

Retrieved evidence:
{context}

Produce the literature review."""


class LiteratureReviewerAgent(BaseAgent):
    name = "literature_reviewer"
    system_prompt = SYSTEM

    def __init__(self, rag: RAGEngine):
        super().__init__()
        self.rag = rag
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", USER),
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

    def run(self, topic: str, top_k: int = 8) -> Dict[str, Any]:
        hits = self.rag.retrieve(topic, top_k=top_k)
        if not hits:
            return {"review": "No evidence in knowledge base.", "sources": []}
        context = self._format(hits)
        chain = self.prompt | self.llm
        resp = chain.invoke({"topic": topic, "context": context})
        sources = sorted({h["metadata"].get("paper_id", "?") for h in hits})
        return {
            "review": resp.content,
            "sources": sources,
            "evidence_count": len(hits),
        }