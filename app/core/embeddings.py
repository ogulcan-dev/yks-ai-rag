import os
from typing import List
from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        # Load the model
        # Using CPU by default as requested (no GPU constraint mentioned, but implied lightweight/no local LLM)
        # However, embeddings run locally.
        self.model = SentenceTransformer(model_name)

    def get_query_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        Prefix with 'query: ' as required by e5 models.
        """
        return self.model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def get_passage_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents/passages.
        Prefix with 'passage: ' as required by e5 models.
        """
        formatted_texts = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(formatted_texts, normalize_embeddings=True, show_progress_bar=True)
        return embeddings.tolist()
