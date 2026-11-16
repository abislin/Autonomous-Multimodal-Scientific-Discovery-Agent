# Autonomous Multimodal Scientific Discovery Agent

An end-to-end, multi-agent AI system that autonomously collects scientific literature, generates hypotheses, plans and executes machine learning experiments, analyzes scientific figures using Vision-Language Models (VLMs), validates results statistically, and synthesizes a complete research paper in LaTeX.

> **⚠️ Disclaimer:** This system generates **AI-assisted research drafts**. The findings, statistics, and papers produced by this system are for demonstration and research assistance purposes only. They must undergo independent scientific validation and peer review before being considered verified scientific discoveries.

---

## 📖 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Multi-Agent Workflow](#-multi-agent-workflow)
3. [Project Structure](#-project-structure)
4. [Prerequisites](#️-prerequisites)
5. [Installation & Setup](#-installation--setup)
6. [Usage](#-usage)
7. [Testing & Validation](#-testing--validation)
8. [Technology Stack](#️-technology-stack)

---

## 🏗️ System Architecture

The project is built over four distinct phases, orchestrated by **LangGraph**:

1. **Week 1 — Foundation:** Automated paper ingestion (arXiv/PubMed), PDF processing (PyMuPDF), ChromaDB vector storage, RAG, Literature Reviewer, and Hypothesis Generator.
2. **Week 2 — Experimentation:** Strict JSON Experiment Planner, isolated Docker sandbox execution, Code Executor with self-debugging feedback loop (RL interface).
3. **Week 3 — Analysis & Validation:** Multimodal figure extraction, VLM integration (LLaVA/GPT-4o), Critic Agent (data leakage & methodology checks), Statistical validation (bootstrap CIs, p-values), and Long-term Research Memory.
4. **Week 4 — Synthesis & Delivery:** Research Synthesis Agent, automated LaTeX paper generation, PDF compilation, Streamlit monitoring dashboard, and holdout evaluation.

---

## 🔄 Multi-Agent Workflow

The graph progresses through these nodes sequentially via `orchestration/workflow.py`:

1. `ingest` ➔ Collects papers from arXiv & PubMed into ChromaDB.
2. `review` ➔ Literature Reviewer Agent summarizes evidence.
3. `hypothesize` ➔ Hypothesis Generator Agent proposes testable hypotheses.
4. `plan` ➔ Experiment Planner Agent creates a strict JSON experiment plan.
5. `execute` ➔ Code Executor Agent generates Python code, runs it in a Docker sandbox, and self-debugs errors.
6. `figures` ➔ Extracts images from PDFs and analyzes them using a VLM.
7. `stats` ➔ Calculates confidence intervals and p-values.
8. `critique` ➔ Critic Agent evaluates methodology and data leakage.
9. `remember` ➔ Saves the outcome to long-term research memory.
10. `collect` ➔ Gathers all context for final synthesis.
11. `synthesize` ➔ Research Synthesis Agent drafts the paper text.
12. `latex` ➔ Jinja2 generator creates `.tex` file, `pdflatex` compiles it to PDF.

---

## 📂 Project Structure

```text
scientific_discovery_agent/
├── main.py                     # CLI entry point
├── config.py                   # Environment configurations
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .env.example                # Example environment variables
│
├── agents/                     # AI Agents
│   ├── base_agent.py           # Common LLM scaffold
│   ├── literature_reviewer.py
│   ├── hypothesis_generator.py
│   ├── experiment_planner.py
│   ├── code_executor.py        # Self-debugging loop
│   ├── critic_agent.py         # Methodology & leakage checks
│   └── synthesis_agent.py      # Bridge to synthesis module
│
├── orchestration/              
│   └── workflow.py             # LangGraph pipeline
│
├── ingestion/                  # Data pipelines
│   ├── arxiv_client.py
│   ├── pubmed_client.py
│   ├── pdf_processor.py
│   └── pipeline.py
│
├── knowledge/                  # RAG & Vector DB
│   ├── embeddings.py
│   ├── vector_store.py         # ChromaDB wrapper
│   └── rag.py
│
├── execution/                  # Docker Sandbox
│   ├── sandbox.py              # Resource/Network policy
│   └── docker_executor.py      # Host-side client
│
├── multimodal/                 # Vision-Language Models
│   ├── figure_extractor.py     # PyMuPDF image extraction
│   ├── chart_classifier.py     # Heuristic chart types
│   └── vlm_client.py           # LLaVA/GPT-4o API wrapper
│
├── statistics/                 # Statistical Validation
│   └── validation.py           # Bootstrap CIs, paired t-tests
│
├── memory/                     # Long-term memory
│   └── research_memory.py      # Chroma-backed memory
│
├── synthesis/                  # Paper Generation
│   ├── synthesis_agent.py      # LLM prompt for drafting
│   ├── latex_generator.py      # Jinja2 rendering
│   ├── latex_compiler.py       # pdflatex wrapper
│   └── templates/
│       └── paper.tex.j2
│
├── dashboard/                  # Streamlit UI
│   └── app.py
│
├── evaluation/                 # System Evaluation
│   └── evaluator.py            # Precision@K, holdout testing
│
├── docker/                     # Sandbox definition
│   ├── Dockerfile
│   └── runner.py
│
└── artifacts/                  # Generated code, logs, figures, PDFs (gitignored)