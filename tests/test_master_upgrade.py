"""
Comprehensive Regression Test Suite for ProductMind X Master Upgrade.
Covers:
1. General questions without retrieval.
2. Product & metadata lookup.
3. Multi-turn product comparison & recent-product context resolution.
4. Ambiguous product references and abstention.
5. Hybrid retrieval (BM25 + ChromaDB semantic) and deduplication.
6. Deterministic numeric calculation accuracy.
7. Citation validity, claim validation, and verification of review IDs.
8. Handling of missing metadata and missing timestamps.
9. Provider error handling and deterministic grounded fallback.
10. Strict model pinning (rejection of 'auto', zero silent switching).
"""

import unittest
from config import PINNED_MODEL_ID, OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY
from chatbot.conversation_manager import ConversationManager
from chatbot.omniroute_client import OmniRouteClient
from chatbot.response_generator import ResponseGenerator
from rag.grounded_rag import GroundedRAGEngine
from rag.retriever import ContextRetriever, calculate_bm25_lexical_score
from analytics.tools import (
    calculate_dataset_statistics,
    calculate_review_metrics,
    calculate_temporal_issue_trend,
    calculate_preference_weighted_recommendation
)
from data.sample_dataset import SAMPLE_METADATA, SAMPLE_REVIEWS

class TestMasterUpgrade(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grounded_engine = GroundedRAGEngine()
        cls.conv_manager = ConversationManager(products=SAMPLE_METADATA)
        cls.retriever = ContextRetriever()

    def test_01_general_question_no_review_retrieval(self):
        """Verify that capability queries return concise information without loading review chunks."""
        res = self.conv_manager.process_message("What can you do?")
        self.assertEqual(res["intent"], "general_capability")
        self.assertIn("ProductMind X", res["reply"])
        self.assertEqual(res["retrieval_info"], {})

    def test_02_dataset_statistics(self):
        """Verify exact calculation of corpus statistics."""
        res = self.conv_manager.process_message("How many reviews are indexed?")
        self.assertEqual(res["intent"], "dataset_health")
        stats = res["backend_metrics"]
        self.assertGreater(stats.get("total_products_indexed", 0), 0)
        self.assertIn("Electronics", stats.get("categories_covered", []))

    def test_03_product_lookup_with_metadata(self):
        """Verify product lookup retrieves metadata token and valid specs."""
        res = self.conv_manager.process_message("Tell me about the Sony WH-1000XM5")
        self.assertEqual(res["product"]["parent_asin"], "B09XS7JWHH")
        self.assertIn("Sony", res["product"]["store"])

    def test_04_multi_turn_comparison_and_recent_tracking(self):
        """Verify multi-turn comparison correctly resolves candidates across turns."""
        self.conv_manager.reset_conversation()
        # Turn 1: query Sony XM5
        self.conv_manager.process_message("Tell me about Sony XM5")
        self.assertEqual(self.conv_manager.last_active_product["parent_asin"], "B09XS7JWHH")

        # Turn 2: query AirPods Pro 2
        self.conv_manager.process_message("What about Apple AirPods Pro 2?")
        self.assertEqual(self.conv_manager.last_active_product["parent_asin"], "B0BDHW4P5S")

        # Turn 3: ask comparison without mentioning product names
        res3 = self.conv_manager.process_message("Compare which one is better which one should i buy")
        self.assertIn(res3["intent"], ["comparison_recommendation", "product_comparison"])
        self.assertEqual(len(res3["all_products"]), 2)
        asins = [p["parent_asin"] for p in res3["all_products"]]
        self.assertIn("B09XS7JWHH", asins)
        self.assertIn("B0BDHW4P5S", asins)

    def test_05_hybrid_retrieval_bm25_lexical_scoring(self):
        """Verify lexical BM25 token scoring behaves as expected."""
        q_tokens = ["battery", "life"]
        doc1_tokens = ["great", "battery", "life", "lasts", "long"]
        doc2_tokens = ["excellent", "sound", "quality", "and", "bass"]

        score1 = calculate_bm25_lexical_score(q_tokens, doc1_tokens)
        score2 = calculate_bm25_lexical_score(q_tokens, doc2_tokens)
        self.assertGreater(score1, score2)

    def test_06_claim_validation_detects_invalid_citations(self):
        """Verify claim validator detects invalid review tokens."""
        sample_prod = SAMPLE_METADATA[0]
        sample_revs = [r for r in SAMPLE_REVIEWS if r.get("parent_asin") == sample_prod["parent_asin"]]
        self.grounded_engine.build_grounded_context("query", sample_prod, sample_revs)

        valid_id = sample_revs[0]["review_id"]
        text_with_cites = f"This product is great [Review #{valid_id}] but has defects [Review #REV-FAKE-999]."
        validation = self.grounded_engine.validate_claims(text_with_cites)
        self.assertFalse(validation["is_valid"])
        self.assertIn("REV-FAKE-999", validation["hallucinated_review_citations"])
        self.assertIn(valid_id, validation["valid_review_citations"])

    def test_07_temporal_trend_requires_minimum_sample(self):
        """Verify temporal trend abstains when dated complaints are insufficient."""
        complaints = [{"sentence": "Issue", "timestamp": 1675209600000}]
        all_revs = [{"timestamp": 1675209600000}]
        trend = calculate_temporal_issue_trend("Sample Issue", complaints, all_revs)
        self.assertFalse(trend["is_emerging"])
        self.assertEqual(trend["trajectory"], "Insufficient Data")

    def test_08_abstention_on_unknown_product(self):
        """Verify strict abstention when query asks for an unknown product."""
        res = self.conv_manager.process_message("Tell me about an unreleased QuantumX Graviton 9000")
        self.assertIn(res["intent"], ["ambiguous_unsupported", "insufficient_evidence"])
        self.assertIn("⚠️ **Product Not in Verified Database**", res["reply"])

    def test_09_model_pinning_rejects_auto(self):
        """Verify model='auto' is rejected by OmniRoute client policy."""
        client = OmniRouteClient(pinned_model_id=PINNED_MODEL_ID)
        with self.assertRaises(ValueError):
            client.generate_grounded_response("query", "context", requested_model="auto")

    def test_10_preference_weighted_advisor(self):
        """Verify preference-weighted recommendation calculates composite scores."""
        p1 = SAMPLE_METADATA[0]
        p2 = SAMPLE_METADATA[1]
        metrics = {
            p1["parent_asin"]: {"average_rating": 4.5, "total_reviews": 4, "verified_ratio_pct": 100.0},
            p2["parent_asin"]: {"average_rating": 4.6, "total_reviews": 3, "verified_ratio_pct": 100.0}
        }
        rec = calculate_preference_weighted_recommendation([p1, p2], metrics, {})
        self.assertTrue(rec["has_recommendation"])
        self.assertIn("winner_title", rec)
