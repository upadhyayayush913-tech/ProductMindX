"""
Conversational Memory with Semantic Recall.
Stores and retrieves conversation turns using ChromaDB vector similarity
to preserve context across multi-turn interactions.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
from config import CHROMA_COLLECTION_MEMORY
from rag.vector_store import ChromaVectorStore
from utils.logger import get_logger

logger = get_logger("memory")

class ConversationMemory:
    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()
        self.coll = self.vector_store.memory_coll

    def save_turn(self, session_id: str, user_query: str, assistant_reply: str, metadata: Optional[Dict[str, Any]] = None):
        """Save a single dialogue turn to memory."""
        turn_id = f"turn_{session_id}_{uuid.uuid4().hex[:8]}"
        content = f"User asked: {user_query}\nAssistant replied: {assistant_reply}"
        vector = self.vector_store.embedder.embed_text(content)

        meta = {
            "session_id": session_id,
            "user_query": user_query[:200],
            "timestamp": time.time(),
            "doc_type": "memory_turn"
        }
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v

        try:
            self.coll.upsert(
                ids=[turn_id],
                documents=[content],
                embeddings=[vector],
                metadatas=[meta]
            )
            logger.debug(f"Saved memory turn {turn_id} for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to save conversation turn to vector store: {e}")

    def recall_relevant_history(self, current_query: str, session_id: Optional[str] = None, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve previous dialogue turns semantically relevant to current query."""
        where = {"session_id": session_id} if session_id else None
        try:
            hits = self.vector_store.search(
                collection_name=CHROMA_COLLECTION_MEMORY,
                query=current_query,
                top_k=top_k,
                where=where
            )
            return hits
        except Exception as e:
            logger.warning(f"Error recalling memory: {e}")
            return []
