from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from config import Config


class BaseAgent(ABC):
    """Common scaffold for all research agents."""

    name: str = "base_agent"
    system_prompt: str = "You are a helpful research assistant."

    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=Config.LLM_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY,
            base_url=Config.OPENAI_BASE_URL,
        )

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        ...