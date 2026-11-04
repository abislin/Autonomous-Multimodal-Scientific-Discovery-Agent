from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import Config
from knowledge.vector_store import VectorStore

RAG_SYSTEM = """You are a scientific research assistant. Answer ONLY using the
provided evidence passages. Every factual claim must be followed by a citation
in the form [paper_id]. Clearly distinguish peer-reviewed publications from
preprints. If the evidence is insufficient, say so explicitly."""

RAG_USER = """Research question: {question}

Evidence passages:
{context}

Write an evidence-grounded summary with inline citations [paper_id]."""


class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vs = vector_store
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM),
            ("human", RAG_USER),
        ])

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        blocks = []
        for h in hits:
            m = h["metadata"]
            tag = "PREPRINT" if m.get("is_preprint") else "PEER-REVIEWED"
            blocks.append(
                f"[{m.get('paper_id','?')}] ({tag}) {m.get('title','')} "
                f"({m.get('source','')}, {m.get('published','')})\n{h['text']}"
            )
        return "\n\n---\n\n".join(blocks)

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        return self.vs.query(question, top_k=top_k)

    def answer(self, question: str, top_k: int = None) -> Dict[str, Any]:
        hits = self.retrieve(question, top_k=top_k)
        if not hits:
            return {"answer": "No evidence found in the knowledge base.",
                    "sources": []}
        context = self._format_context(hits)
        chain = self.prompt | self.llm
        resp = chain.invoke({"question": question, "context": context})
        sources = sorted({h["metadata"].get("paper_id", "?") for h in hits})
        return {"answer": resp.content, "sources": sources, "hits": hits}