from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            print(f"Loading the model {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print("Model loaded successfully!")
        except:
            print("Error loading model!")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        try:
            print("Generating embeddings...")
            results = self._model.encode(texts)
            print(f"Embeddings generated! shape: {results.shape}")
            return results
        except:
            print("Error generating embeddings!")
            raise


_embedding_manager = None


def get_embedding_manager() -> EmbeddingManager:
    global _embedding_manager
    if _embedding_manager is None:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager
