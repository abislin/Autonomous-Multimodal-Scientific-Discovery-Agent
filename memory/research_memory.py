import hashlib
import json
import time
from typing import Dict, Any, List, Optional
from knowledge.embeddings import get_embedding_function
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import Config


MEMORY_COLLECTION = "research_memory"


class ResearchMemory:
    """Persist and retrieve hypotheses, plans, metrics, and findings."""

    def __init__(self):
        Config.ensure_dirs()
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedder = get_embedding_function()
        self.col = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, text: str) -> List[float]:
        return self.embedder.embed_query(text)

    # ---------- write ----------
    def remember(self, kind: str, topic: str, text: str,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """kind ∈ {hypothesis, plan, result, finding, critique}"""
        mem_id = hashlib.sha1(
            f"{kind}:{topic}:{text[:80]}:{time.time()}".encode()
        ).hexdigest()[:16]
        meta = {"kind": kind, "topic": topic,
                "created_at": time.time()}
        if metadata:
            # Chroma only supports scalar metadata
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = json.dumps(v)[:500]
        self.col.upsert(
            ids=[mem_id],
            documents=[text],
            metadatas=[meta],
            embeddings=[self._embed(text)],
        )
        return mem_id

    # ---------- read ----------
    def recall(self, query: str, kind: Optional[str] = None,
               top_k: int = 5) -> List[Dict[str, Any]]:
        where = {"kind": kind} if kind else None
        res = self.col.query(
            query_embeddings=[self._embed(query)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            })
        return out

    def has_similar_experiment(self, topic: str, threshold: float = 0.2
                               ) -> Optional[Dict[str, Any]]:
        hits = self.recall(topic, kind="result", top_k=1)
        if hits and hits[0]["distance"] < threshold:
            return hits[0]
        return None