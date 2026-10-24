"""
Sentence Embedding Provider for RAG.
Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) for 384-dimensional
dense vector representation with graceful fallback.
"""

from typing import List, Union
import numpy as np
from config import EMBEDDING_MODEL_NAME
from utils.logger import get_logger

logger = get_logger("embedder")

class TextEmbedder:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        if self._initialized:
            return
        self.model_name = model_name
        self.model = None
        self._load_model()
        self._initialized = True

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device="cpu")
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using normalized bag-of-words fallback.")
            self.model = None

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string into a vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of strings."""
        if not texts:
            return []

        if self.model is not None:
            try:
                embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
                return [e.tolist() for e in embeddings]
            except Exception as e:
                logger.warning(f"Error during neural embedding inference: {e}")

        # Deterministic lightweight fallback embedding (dimension 384)
        vectors = []
        for text in texts:
            vec = np.zeros(384, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = hash(w) % 384
                vec[h] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec.tolist())
        return vectors
