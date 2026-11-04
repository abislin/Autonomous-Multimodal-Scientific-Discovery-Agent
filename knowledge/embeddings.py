from langchain_community.embeddings import HuggingFaceEmbeddings
from config import Config

def get_embedding_function():
    """Sentence-Transformers based embedding function for Chroma."""
    return HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )