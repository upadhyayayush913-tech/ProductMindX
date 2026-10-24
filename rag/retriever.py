"""
Hybrid Context Retriever with BM25/Keyword Matching, Semantic Search, and Provenance Reranking.
Combines:
1. Semantic dense vector retrieval via ChromaDB.
2. Keyword / lexical BM25 token overlap.
3. Structured metadata filters (ASIN or category).
4. Deduplication and relevance reranking.
5. Supporting and contradictory evidence grouping.
"""

import re
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from config import (
    CHROMA_COLLECTION_PROFILES,
    CHROMA_COLLECTION_INSIGHTS,
    CHROMA_COLLECTION_REVIEWS,
    RETRIEVER_TOP_K
)
from rag.vector_store import ChromaVectorStore
from utils.logger import get_logger

logger = get_logger("retriever")

def tokenize_words(text: str) -> List[str]:
    """Tokenize text into lowercase alpha words."""
    return [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text)]

def calculate_bm25_lexical_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Calculate token overlap / BM25 proxy score."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_freq = Counter(doc_tokens)
    score = 0.0
    for q in query_tokens:
        tf = doc_freq.get(q, 0)
        if tf > 0:
            score += (tf / (tf + 1.2))
    return score

class ContextRetriever:
    def __init__(self, vector_store: Optional[ChromaVectorStore] = None):
        self.vector_store = vector_store or ChromaVectorStore()

    def hybrid_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks by combining semantic similarity distance and lexical token match.
        Deduplicates chunks based on text hash or review ID.
        """
        if not candidates:
            return []

        query_tokens = tokenize_words(query)
        seen_contents: Set[str] = set()
        unique_candidates = []

        for c in candidates:
            content_key = c.get("content", "").strip().lower()[:120]
            if content_key and content_key not in seen_contents:
                seen_contents.add(content_key)
                unique_candidates.append(c)

        scored = []
        for c in unique_candidates:
            # Semantic score from Chroma distance (Chroma uses cosine distance: 0 is exact match, 2 is opposite)
            chroma_dist = c.get("distance", 1.0)
            semantic_score = max(0.0, 1.0 - (chroma_dist / 2.0))

            # Lexical score
            doc_tokens = tokenize_words(c.get("content", ""))
            lexical_score = calculate_bm25_lexical_score(query_tokens, doc_tokens)

            # Combined hybrid score (70% semantic, 30% lexical bonus)
            hybrid_score = round((0.70 * semantic_score) + (0.30 * min(1.0, lexical_score / 3.0)), 4)
            c["hybrid_score"] = hybrid_score
            scored.append(c)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:top_k]

    def retrieve_context(
        self,
        query: str,
        target_asin: Optional[str] = None,
        intent: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = RETRIEVER_TOP_K
    ) -> Dict[str, Any]:
        """
        Execute hybrid retrieval across product profiles, NLP insights, and customer reviews.
        Uses exact single-field ChromaDB filters to guarantee compatibility.
        """
        where_filter = None
        if target_asin:
            where_filter = {"asin": target_asin}
        elif category:
            where_filter = {"category": category}

        # 1. Search profiles
        raw_profiles = self.vector_store.search(
            collection_name=CHROMA_COLLECTION_PROFILES,
            query=query,
            top_k=max(2, top_k // 2),
            where=where_filter
        )
        ranked_profiles = self.hybrid_rerank(query, raw_profiles, top_k=2)

        # 2. Search insights (pros, cons, issues)
        raw_insights = self.vector_store.search(
            collection_name=CHROMA_COLLECTION_INSIGHTS,
            query=query,
            top_k=max(4, top_k),
            where=where_filter
        )
        ranked_insights = self.hybrid_rerank(query, raw_insights, top_k=4)

        # 3. Search raw reviews
        raw_reviews = self.vector_store.search(
            collection_name=CHROMA_COLLECTION_REVIEWS,
            query=query,
            top_k=max(6, top_k * 2),
            where=where_filter
        )
        ranked_reviews = self.hybrid_rerank(query, raw_reviews, top_k=6)

        # Separate into supporting and contradictory evidence when available
        supporting_reviews = [r for r in ranked_reviews if r.get("metadata", {}).get("rating", 4.0) >= 4.0]
        critical_reviews = [r for r in ranked_reviews if r.get("metadata", {}).get("rating", 4.0) <= 3.0]

        # Combine documents into structured context block
        context_parts = []

        if ranked_profiles:
            context_parts.append("--- PRODUCT OVERVIEW & SPECIFICATIONS ---")
            for p in ranked_profiles:
                context_parts.append(p["content"])

        if ranked_insights:
            context_parts.append("\n--- EXTRACTED NLP INSIGHTS (PROS, CONS & PROBLEMS) ---")
            for ins in ranked_insights:
                context_parts.append(ins["content"])

        if ranked_reviews:
            context_parts.append("\n--- RELEVANT CUSTOMER REVIEWS & TESTIMONIALS ---")
            for r in ranked_reviews:
                context_parts.append(f"- {r['content']}")

        full_context_text = "\n".join(context_parts)

        return {
            "query": query,
            "target_asin": target_asin,
            "intent": intent,
            "formatted_context": full_context_text,
            "profiles_retrieved": len(ranked_profiles),
            "insights_retrieved": len(ranked_insights),
            "reviews_retrieved": len(ranked_reviews),
            "supporting_evidence_count": len(supporting_reviews),
            "contradictory_evidence_count": len(critical_reviews),
            "raw_hits": {
                "profiles": ranked_profiles,
                "insights": ranked_insights,
                "reviews": ranked_reviews
            }
        }
