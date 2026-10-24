"""
Vector Database Store using ChromaDB.
Manages persistent semantic collections:
- product_profiles
- product_insights (Pros, Cons, Issues, Topics)
- review_chunks
- conversation_memory
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import chromadb
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION_PROFILES,
    CHROMA_COLLECTION_INSIGHTS,
    CHROMA_COLLECTION_REVIEWS,
    CHROMA_COLLECTION_MEMORY
)
from rag.embedder import TextEmbedder
from utils.logger import get_logger

logger = get_logger("vector_store")

class ChromaVectorStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ChromaVectorStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, persist_dir: Path = CHROMA_DIR):
        if self._initialized:
            return
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = TextEmbedder()

        logger.info(f"Initializing ChromaDB PersistentClient at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        # Get or create collections
        self.profiles_coll = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_PROFILES,
            metadata={"description": "Product metadata and descriptions"}
        )
        self.insights_coll = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_INSIGHTS,
            metadata={"description": "Extracted Pros, Cons, and Emerging Issues"}
        )
        self.reviews_coll = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_REVIEWS,
            metadata={"description": "Individual review excerpts"}
        )
        self.memory_coll = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_MEMORY,
            metadata={"description": "Conversation turns for contextual recall"}
        )
        self._initialized = True

    def add_product_profile(self, asin: str, title: str, category: str, content: str, metadata: Dict[str, Any]):
        """Add product profile document."""
        doc_id = f"profile_{asin}"
        vector = self.embedder.embed_text(content)
        # Ensure metadata values are str, int, float, or bool for ChromaDB
        clean_meta = {
            "asin": asin,
            "title": str(title),
            "category": str(category),
            "doc_type": "profile"
        }
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = str(v)

        self.profiles_coll.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[vector],
            metadatas=[clean_meta]
        )

    def add_product_insight(self, asin: str, title: str, insight_type: str, content: str, extra_meta: Optional[Dict[str, Any]] = None):
        """Add insight document (pros, cons, issues, or topics)."""
        doc_id = f"insight_{asin}_{insight_type}"
        vector = self.embedder.embed_text(content)
        meta = {
            "asin": asin,
            "title": str(title),
            "doc_type": insight_type
        }
        if extra_meta:
            for k, v in extra_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)

        self.insights_coll.upsert(
            ids=[doc_id],
            documents=[content],
            embeddings=[vector],
            metadatas=[meta]
        )

    def add_review_chunks(self, asin: str, title: str, chunks: List[Dict[str, Any]]):
        """Add review excerpt chunks."""
        if not chunks:
            return
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = f"rev_{asin}_{i}"
            text = chunk.get("text", "")
            if not text.strip():
                continue
            ids.append(doc_id)
            documents.append(text)
            metadatas.append({
                "asin": asin,
                "title": str(title),
                "rating": float(chunk.get("rating", 3.0)),
                "verified": bool(chunk.get("verified_purchase", True)),
                "doc_type": "review",
                "review_id": str(chunk.get("review_id", f"REV-{asin[:6]}-{i}")),
                "translated": bool(chunk.get("translated", False)),
                "language": str(chunk.get("original_language", chunk.get("language", "en"))),
                "original_text": str(chunk.get("original_text", text))[:300]
            })

        if documents:
            vectors = self.embedder.embed_batch(documents)
            self.reviews_coll.upsert(
                ids=ids,
                documents=documents,
                embeddings=vectors,
                metadatas=metadatas
            )

    def search(self, collection_name: str, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform semantic vector similarity search against a specific collection."""
        collection_map = {
            CHROMA_COLLECTION_PROFILES: self.profiles_coll,
            CHROMA_COLLECTION_INSIGHTS: self.insights_coll,
            CHROMA_COLLECTION_REVIEWS: self.reviews_coll,
            CHROMA_COLLECTION_MEMORY: self.memory_coll
        }
        coll = collection_map.get(collection_name)
        if coll is None:
            logger.warning(f"Collection {collection_name} not found.")
            return []

        try:
            count = coll.count()
            if count == 0:
                return []
            k = min(top_k, count)
            query_vector = self.embedder.embed_text(query)

            results = coll.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=where
            )

            hits = []
            if results and results.get("documents"):
                docs = results["documents"][0]
                metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                dists = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
                ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

                for doc_id, doc, meta, dist in zip(ids, docs, metas, dists):
                    # Chroma distances are L2 or cosine distance; convert to similarity score ~ 0 to 1
                    score = max(0.0, 1.0 - (dist / 2.0))
                    hits.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                        "similarity_score": round(score, 3)
                    })

            return hits
        except Exception as e:
            logger.error(f"Error querying ChromaDB collection {collection_name}: {e}")
            return []
