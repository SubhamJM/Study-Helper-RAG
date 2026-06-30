from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loadint the model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully!")
        except:
            print("Error loading model!")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            if not self.model:
                raise ValueError("model not loaded")
            print("Generating Embeddings...")
            results = self.model.encode(texts)
            print("embeddings generated successfully!")
            print(f"shape of the embeddings: {results.shape}")
            return results
        except:
            print("Error generating embeddings!")
            return []

embedding_manager = EmbeddingManager()