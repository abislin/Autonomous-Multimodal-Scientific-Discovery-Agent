import uuid
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from agents.base_agent import BaseAgent
from execution.docker_executor import DockerExecutor
from storage.experiment_store import ExperimentStore

SYSTEM = """You are a Code Executor Agent. Generate a single, self-contained
Python script that implements the experiment plan.

HARD RULES:
- The script MUST define `main() -> dict` returning metric name → float.
- Allowed imports ONLY: numpy, pandas, sklearn, scipy, math, statistics.
- NO file I/O, NO network, NO subprocess, NO os/sys (except `if __name__`).
- Use sklearn.datasets for datasets; no downloads.
- Seed everything with random_state=42.
- Print nothing except what sklearn may print; the runner captures main()'s return.
- Return ONLY the Python code (no markdown fences, no prose).
"""

USER = """Experiment plan (JSON):
{plan}

Generate the complete Python script.

{error_block}"""

ERROR_BLOCK = """Previous attempt FAILED with:
--- ERROR ---
{error}
--- TRACEBACK ---
{traceback}
--- LOGS (tail) ---
{logs}

Fix the specific bug. Keep the same plan. Return corrected Python only."""


class CodeExecutorAgent(BaseAgent):
    name = "code_executor"
    system_prompt = SYSTEM
    MAX_ATTEMPTS = 4

    def __init__(self, docker_executor: DockerExecutor,
                 store: ExperimentStore):
        super().__init__()
        self.executor = docker_executor
        self.store = store
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM), ("human", USER),
        ])

    def _strip_fences(self, code: str) -> str:
        code = code.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1] if "\n" in code else code
            if code.rstrip().endswith("```"):
                code = code.rstrip()[:-3]
        return code

    def run(self, experiment_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        plan_str = __import__("json").dumps(plan, indent=2)
        last = {}
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            error_block = ""
            if last and not last.get("ok"):
                error_block = ERROR_BLOCK.format(
                    error=last.get("error") or "",
                    traceback=(last.get("traceback") or "")[-4000:],
                    logs=(last.get("logs") or "")[-2000:],
                )
            chain = self.prompt | self.llm
            resp = chain.invoke({
                "plan": plan_str,
                "error_block": error_block,
            })
            code = self._strip_fences(resp.content)

            record = self.executor.run_code(
                code, run_id=f"{experiment_id}_a{attempt}"
            )
            self.store.log_run(experiment_id, attempt, code, record)

            last = record
            if record["ok"]:
                return {
                    "ok": True,
                    "attempts": attempt,
                    "code": code,
                    "metrics": record["metrics"],
                    "run_id": record["run_id"],
                }

        return {
            "ok": False,
            "attempts": self.MAX_ATTEMPTS,
            "code": last.get("code", ""),
            "error": last.get("error"),
            "metrics": {},
            "run_id": last.get("run_id"),
        }