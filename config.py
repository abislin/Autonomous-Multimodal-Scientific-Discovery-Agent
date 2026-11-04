import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = 0.2

    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Storage
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
    PDF_DIR = os.getenv("PDF_DIR", "./pdfs")
    COLLECTION_NAME = "scientific_papers"

    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150

    # Retrieval
    TOP_K = 8

    # arXiv / PubMed
    ARXIV_MAX_RESULTS = 15
    PUBMED_MAX_RESULTS = 15

    @classmethod
    def ensure_dirs(cls):
        Path(cls.CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        Path(cls.PDF_DIR).mkdir(parents=True, exist_ok=True)