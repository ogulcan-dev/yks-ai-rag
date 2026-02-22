from functools import lru_cache
import logging
from app.core.vectorstore import VectorStore
from app.core.embeddings import EmbeddingModel
from app.core.llm import LLMClient

logger = logging.getLogger(__name__)

@lru_cache()
def get_vector_store() -> VectorStore:
    store = VectorStore()
    if not store.load():
        logger.warning("Vector index not found.")
    return store

@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()

@lru_cache()
def get_llm_client() -> LLMClient | None:
    try:
        return LLMClient()
    except Exception as e:
        logger.warning(f"LLM Client init failed: {e}")
        return None
