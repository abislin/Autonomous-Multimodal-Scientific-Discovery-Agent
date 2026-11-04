import json
import time
import uuid
from pathlib import Path
from typing import Dict, Any
import docker
from docker.errors import ContainerError, DockerException
from execution.sandbox import SandboxPolicy
from config import Config


class DockerExecutor:
    def __init__(self, policy: SandboxPolicy = None):
        self.policy = policy or SandboxPolicy()
        self.client = docker.from_env()
        self.artifacts = Path("artifacts")
        (self.artifacts / "code").mkdir(parents=True, exist_ok=True)
        (self.artifacts / "logs").mkdir(parents=True, exist_ok=True)
        (self.artifacts / "results").mkdir(parents=True, exist_ok=True)

    def run_code(self, code: str, run_id: str = None) -> Dict[str, Any]:
        run_id = run_id or uuid.uuid4().hex[:10]
        workdir = self.artifacts / "code" / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "experiment.py").write_text(code)

        container = None
        started = time.time()
        try:
            container = self.client.containers.run(
                **self.policy.docker_kwargs(str(workdir.resolve()))
            )
            timed_out = False
        except DockerException as e:
            return {
                "run_id": run_id, "ok": False,
                "error": f"DockerException: {e}", "traceback": None,
                "metrics": {}, "duration_s": time.time() - started,
                "logs": "", "timed_out": False,
            }

        # Enforce wall-clock timeout on the running container
        deadline = started + self.policy.timeout_s
        while True:
            try:
                container.reload()
                if container.status in ("exited", "dead"):
                    break
            except DockerException:
                break
            if time.time() > deadline:
                try: container.kill()
                except Exception: pass
                break
            time.sleep(1)

        timed_out = time.time() > deadline
        try:
            logs = container.logs(stdout=True, stderr=True).decode(errors="replace")
        except Exception:
            logs = ""
        try:
            container.remove(force=True)
        except Exception:
            pass

        # Persist logs
        (self.artifacts / "logs" / f"{run_id}.log").write_text(logs)

        result = {"ok": False, "metrics": {}, "error": None, "traceback": None}
        for line in logs.splitlines():
            if line.startswith("__RESULT__"):
                try: result = json.loads(line[len("__RESULT__"):])
                except json.JSONDecodeError: pass
                break

        if timed_out and result["error"] is None:
            result["error"] = f"TimeoutError: exceeded {self.policy.timeout_s}s"

        record = {
            "run_id": run_id,
            "ok": bool(result["ok"]),
            "metrics": result.get("metrics", {}),
            "error": result.get("error"),
            "traceback": result.get("traceback"),
            "logs": logs[-8000:],
            "duration_s": round(time.time() - started, 2),
            "timed_out": timed_out,
        }
        (self.artifacts / "results" / f"{run_id}.json").write_text(
            json.dumps(record, indent=2)
        )
        return record