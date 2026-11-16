import json
import time
from pathlib import Path

import streamlit as st

from config import Config
from storage.experiment_store import ExperimentStore
from memory.research_memory import ResearchMemory


st.set_page_config(
    page_title="Autonomous Scientific Discovery Agent",
    layout="wide",
)


@st.cache_resource
def get_store():
    return ExperimentStore()


@st.cache_resource
def get_memory():
    return ResearchMemory()


store = get_store()
memory = get_memory()

st.title("Autonomous Multimodal Scientific Discovery Agent")
st.caption("Monitoring dashboard — Week 4")

# ---------- Sidebar: experiment selector ----------
cur = store.conn.execute(
    "SELECT id, topic, created_at FROM experiments ORDER BY created_at DESC LIMIT 50"
)
rows = cur.fetchall()
exp_ids = [r[0] for r in rows]

if not exp_ids:
    st.warning("No experiments yet. Run `python main.py discover --topic \"...\"` first.")
    st.stop()

labels = [f"{r[0]} — {r[1][:60]}" for r in rows]
selected = st.sidebar.selectbox("Select experiment", labels)
exp_id = exp_ids[labels.index(selected)]

# ---------- Tabs ----------
tab_overview, tab_lit, tab_hyp, tab_exp, tab_res, tab_critic, tab_mem, tab_paper = st.tabs([
    "Overview", "Literature", "Hypotheses", "Experiments",
    "Results", "Critic", "Research Memory", "Paper",
])

# ---------- Overview ----------
with tab_overview:
    st.subheader("Overview")
    row = store.conn.execute(
        "SELECT topic, hypothesis, created_at FROM experiments WHERE id=?",
        (exp_id,),
    ).fetchone()
    if row:
        st.markdown(f"**Topic:** {row[0]}")
        st.markdown(f"**Created:** {time.ctime(row[2])}")
        st.markdown("**Chosen hypothesis (truncated):**")
        st.code((row[1] or "")[:800])

    st.markdown("### Workflow progress")
    progress = {
        "ingest": True,
        "review": True,
        "hypothesize": bool(row and row[1]),
        "plan": bool(store.conn.execute(
            "SELECT 1 FROM experiments WHERE id=? AND plan_json IS NOT NULL",
            (exp_id,)).fetchone()),
        "execute": bool(store.conn.execute(
            "SELECT 1 FROM runs WHERE experiment_id=? AND ok=1",
            (exp_id,)).fetchone()),
        "figures": bool(store.conn.execute(
            "SELECT 1 FROM figures WHERE experiment_id=?", (exp_id,)).fetchone()),
        "stats": bool(store.conn.execute(
            "SELECT 1 FROM stats_reports WHERE experiment_id=?",
            (exp_id,)).fetchone()),
        "critique": bool(store.conn.execute(
            "SELECT 1 FROM critiques WHERE experiment_id=?",
            (exp_id,)).fetchone()),
        "paper": bool(store.latest_paper(exp_id)),
    }
    for step, done in progress.items():
        st.write(f"{'✅' if done else '❌'} {step}")

# ---------- Literature ----------
with tab_lit:
    st.subheader("Retrieved scientific papers")
    synth = store.latest_synthesis(exp_id)
    refs = (synth or {}).get("references", [])
    if not refs:
        st.info("No synthesis references yet.")
    else:
        for r in refs:
            st.markdown(
                f"**[{r.get('paper_id')}]** {r.get('title')}  \n"
                f"*{r.get('authors')}* — {r.get('venue')} ({r.get('year')})"
            )

# ---------- Hypotheses ----------
with tab_hyp:
    st.subheader("Generated hypotheses")
    row = store.conn.execute(
        "SELECT hypothesis FROM experiments WHERE id=?", (exp_id,)
    ).fetchone()
    if row:
        st.text(row[0])
    if synth:
        st.markdown("**Synthesis abstract:**")
        st.write(synth.get("abstract", ""))

# ---------- Experiments ----------
with tab_exp:
    st.subheader("Experiment runs")
    cur = store.conn.execute(
        "SELECT attempt, ok, metrics_json, error, duration_s, created_at "
        "FROM runs WHERE experiment_id=? ORDER BY id", (exp_id,)
    )
    run_rows = cur.fetchall()
    if not run_rows:
        st.info("No runs.")
    else:
        for r in run_rows:
            st.write({
                "attempt": r[0],
                "ok": bool(r[1]),
                "metrics": json.loads(r[2] or "{}"),
                "error": r[3],
                "duration_s": r[4],
                "at": time.ctime(r[5]),
            })

# ---------- Results ----------
with tab_res:
    st.subheader("Metrics")
    history = store.history(exp_id)
    for h in history:
        if h["ok"]:
            for k, v in h["metrics"].items():
                st.metric(label=k, value=f"{v:.4f}" if isinstance(v, float) else v)

# ---------- Critic ----------
with tab_critic:
    st.subheader("Critic Agent report")
    row = store.conn.execute(
        "SELECT report_json FROM critiques WHERE experiment_id=? "
        "ORDER BY id DESC LIMIT 1", (exp_id,)
    ).fetchone()
    if not row:
        st.info("No critique yet.")
    else:
        st.json(json.loads(row[0]))

# ---------- Research Memory ----------
with tab_mem:
    st.subheader("Research memory")
    q = st.text_input("Recall query", value="")
    if q:
        hits = memory.recall(q, top_k=5)
        for h in hits:
            st.write({
                "distance": h["distance"],
                "metadata": h["metadata"],
                "text_head": h["text"][:300],
            })

# ---------- Paper ----------
with tab_paper:
    st.subheader("Generated research paper")
    paper = store.latest_paper(exp_id)
    if not paper:
        st.info("No paper generated yet.")
    else:
        st.write({"status": paper["status"],
                  "latex": paper["latex_path"],
                  "pdf": paper["pdf_path"]})
        if paper.get("pdf_path") and Path(paper["pdf_path"]).exists():
            with open(paper["pdf_path"], "rb") as f:
                st.download_button(
                    "Download PDF", f,
                    file_name=Path(paper["pdf_path"]).name,
                    mime="application/pdf",
                )
        tex = Path(paper["latex_path"]) if paper.get("latex_path") else None
        if tex and tex.exists():
            with st.expander("View LaTeX source"):
                st.code(tex.read_text(encoding="utf-8")[:8000], language="latex")