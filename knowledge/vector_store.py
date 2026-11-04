from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings
from config import Config
from knowledge.embeddings import get_embedding_function

class VectorStore:
    """ChromaDB wrapper for scientific passages with metadata."""

    def __init__(self):
        Config.ensure_dirs()
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedder = get_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(texts)

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Each chunk: {id, text, metadata}
        metadata should contain: paper_id, title, authors, source, published,
                                 page, is_preprint, url
        """
        if not chunks:
            return
        ids = [c["id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        embeddings = self._embed(texts)

        # Chroma upsert handles duplicates
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, top_k: int = None) -> List[Dict[str, Any]]:
        top_k = top_k or Config.TOP_K
        q_emb = self.embedder.embed_query(query_text)
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def count(self) -> int:
        return self.collection.count()