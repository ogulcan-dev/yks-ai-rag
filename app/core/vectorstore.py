import logging
import os
import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, index_path: str = "index/faiss.index", metadata_path: str = "index/metadata.pkl"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []  # List of dicts, corresponding to index IDs

    def create_index(self, dimension: int):
        """
        Create a new FAISS index for cosine similarity (Inner Product on normalized vectors).
        """
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []

    def add_documents(self, embeddings: List[List[float]], documents: List[Dict[str, Any]]):
        """
        Add embeddings and corresponding metadata to the index.
        """
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index or load_index first.")
        
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings and documents must match.")

        # Convert to float32 numpy array
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.metadata.extend(documents)

    def save(self):
        """
        Save index and metadata to disk.
        """
        if self.index is None:
            raise ValueError("No index to save.")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Index saved to {self.index_path}")
        logger.info(f"Metadata saved to {self.metadata_path}")

    def load(self):
        """
        Load index and metadata from disk.
        """
        if not os.path.exists(self.index_path) or not os.path.exists(self.metadata_path):
            logger.warning("Index or metadata file not found.")
            return False

        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        return True

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search the index for the query vector.
        Returns a list of (metadata, score) tuples.
        """
        if self.index is None:
            raise ValueError("Index not loaded.")

        # Reshape for FAISS
        vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((self.metadata[idx], float(dist)))
        
        return results
