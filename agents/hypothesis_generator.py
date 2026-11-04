from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent

SYSTEM = """You are a Hypothesis Generator Agent.
Given a literature review, produce novel, testable research hypotheses.
For each hypothesis return:
- Hypothesis statement
- Rationale (grounded in the review)
- Proposed experimental method (concrete, actionable)
- Evaluation metrics
- Supporting references [paper_id]
Reject hypotheses that merely restate existing findings.
"""

USER = """Research topic: {topic}

Literature review:
{review}

Available source ids: {sources}

Generate 3-5 testable hypotheses in a numbered list, each with the required fields."""


class HypothesisGeneratorAgent(BaseAgent):
    name = "hypothesis_generator"
    system_prompt = SYSTEM

    def __init__(self):
        super().__init__()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", USER),
        ])

    def run(self, topic: str, review: str, sources) -> Dict[str, Any]:
        chain = self.prompt | self.llm
        resp = chain.invoke({
            "topic": topic,
            "review": review,
            "sources": ", ".join(sources) if sources else "(none)",
        })
        return {"hypotheses": resp.content, "source_count": len(sources)}