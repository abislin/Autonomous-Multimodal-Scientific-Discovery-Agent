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

SCHEMA_V3 = """
CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    experiment_id TEXT,
    paper_id TEXT,
    page INTEGER,
    image_path TEXT,
    caption TEXT,
    vlm_json TEXT,
    chart_type TEXT,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS critiques (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    report_json TEXT,
    verdict TEXT,
    severity TEXT,
    methodology_score REAL,
    created_at REAL
);
CREATE TABLE IF NOT EXISTS stats_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT,
    report_json TEXT,
    p_value REAL,
    significant INTEGER,
    created_at REAL
);
"""


class ExperimentStore:
    def __init__(self, path: Path = DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.executescript(SCHEMA)
        self.conn.executescript(SCHEMA_V3)
        self.conn.commit()

    # ---------- experiments ----------
    def create_experiment(self, exp_id: str, topic: str, hypothesis: str,
                          plan: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?)",
            (exp_id, topic, hypothesis, json.dumps(plan), time.time()),
        )
        self.conn.commit()

    # ---------- runs ----------
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

    # ---------- figures (Week 3) ----------
    def save_figure(self, experiment_id: str, fig: Dict[str, Any],
                    vlm_result: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO figures VALUES (?,?,?,?,?,?,?,?,?)",
            (fig["figure_id"], experiment_id, fig["paper_id"], fig["page"],
             fig["image_path"], fig.get("caption"),
             json.dumps(vlm_result),
             (vlm_result or {}).get("chart_type", "unknown"),
             time.time()),
        )
        self.conn.commit()

    # ---------- critiques (Week 3) ----------
    def save_critique(self, experiment_id: str,
                      report: Dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO critiques (experiment_id, report_json, verdict, "
            "severity, methodology_score, created_at) VALUES (?,?,?,?,?,?)",
            (experiment_id, json.dumps(report),
             report.get("verdict"), report.get("severity"),
             float(report.get("methodology_score", 0.0) or 0.0),
             time.time()),
        )
        self.conn.commit()

    # ---------- statistics (Week 3) ----------
    def save_stats(self, experiment_id: str,
                   report: Dict[str, Any]) -> None:
        comp = report.get("comparison", {}) or {}
        self.conn.execute(
            "INSERT INTO stats_reports (experiment_id, report_json, p_value, "
            "significant, created_at) VALUES (?,?,?,?,?)",
            (experiment_id, json.dumps(report),
             float(comp.get("p_value", 1.0) or 1.0),
             int(bool(comp.get("significant", False))),
             time.time()),
        )
        self.conn.commit()