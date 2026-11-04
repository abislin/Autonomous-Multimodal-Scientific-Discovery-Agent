import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List

DB_PATH = Path("artifacts/experiments.sqlite3")

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    topic TEXT,
    hypothesis TEXT,
    plan_json TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    attempt INTEGER,
    code TEXT,
    ok INTEGER,
    metrics_json TEXT,
    error TEXT,
    traceback TEXT,
    logs TEXT,
    duration_s REAL,
    created_at REAL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
"""


class ExperimentStore:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def create_experiment(self, exp_id: str, topic: str, hypothesis: str,
                          plan: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?)",
            (exp_id, topic, hypothesis, json.dumps(plan), time.time()),
        )
        self.conn.commit()

    def log_run(self, exp_id: str, attempt: int, code: str,
                record: Dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO runs
               (experiment_id, attempt, code, ok, metrics_json, error,
                traceback, logs, duration_s, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (exp_id, attempt, code, int(record["ok"]),
             json.dumps(record.get("metrics", {})),
             record.get("error"), record.get("traceback"),
             record.get("logs", ""), record.get("duration_s", 0.0),
             time.time()),
        )
        self.conn.commit()

    def best_run(self, exp_id: str) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT attempt, metrics_json, error FROM runs "
            "WHERE experiment_id=? AND ok=1 ORDER BY id DESC LIMIT 1",
            (exp_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {"attempt": row[0], "metrics": json.loads(row[1]), "error": row[2]}

    def history(self, exp_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT attempt, ok, metrics_json, error, duration_s "
            "FROM runs WHERE experiment_id=? ORDER BY id",
            (exp_id,),
        )
        return [
            {"attempt": r[0], "ok": bool(r[1]), "metrics": json.loads(r[2]),
             "error": r[3], "duration_s": r[4]}
            for r in cur.fetchall()
        ]