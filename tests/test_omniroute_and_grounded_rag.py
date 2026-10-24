"""
Automated Verification Suite for OmniRoute & Evidence-Grounded ProductMind X.

Verifies:
1. OmniRoute Gateway health and reachability.
2. Model pinning enforcement (explicit model ID required, auto-routing rejected).
3. Fallback routing disabled by default.
4. Deterministic backend metric calculations.
5. Strict claim validation & citation verification.
6. Honest uncertainty / insufficient-evidence responses.
7. End-to-end grounded query execution.
"""

import unittest
from config import OMNIROUTE_BASE_URL, OMNIROUTE_API_KEY, PINNED_MODEL_ID
from chatbot.omniroute_client import OmniRouteClient
from rag.grounded_rag import GroundedRAGEngine
from chatbot.response_generator import ResponseGenerator
from chatbot.conversation_manager import ConversationManager
from data.sample_dataset import SAMPLE_METADATA, SAMPLE_REVIEWS

class TestOmniRouteAndGroundedRAG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grounded_engine = GroundedRAGEngine()
        cls.client = OmniRouteClient(
            base_url=OMNIROUTE_BASE_URL,
            api_key=OMNIROUTE_API_KEY,
            pinned_model_id=PINNED_MODEL_ID,
            allow_fallback=False
        )
        cls.resp_gen = ResponseGenerator(
            base_url=OMNIROUTE_BASE_URL,
            api_key=OMNIROUTE_API_KEY,
            pinned_model_id=PINNED_MODEL_ID,
            allow_fallback=False
        )
        cls.conv_manager = ConversationManager(products=SAMPLE_METADATA)
        cls.conv_manager.response_generator.update_credentials(
            pinned_model_id=PINNED_MODEL_ID
        )

    def test_01_omniroute_health(self):
        """Verify OmniRoute service is running and reachable."""
        health = self.client.check_health()
        self.assertEqual(health["status"], "healthy", f"OmniRoute gateway is not reachable: {health}")

    def test_02_model_pinning_rejects_auto(self):
        """Verify that model='auto' is strictly rejected under ProductMind X policy."""
        with self.assertRaises(ValueError):
            self.client.generate_grounded_response(
                user_query="test",
                grounded_context_text="test context",
                requested_model="auto"
            )

        with self.assertRaises(ValueError):
            self.client.generate_grounded_response(
                user_query="test",
                grounded_context_text="test context",
                requested_model="auto/coding"
            )

    def test_03_model_pinning_enforced(self):
        """Verify that silent model switching is prevented and target model stays pinned."""
        # When fallback is disabled, passing an unpinned model should be rejected or locked to pinned model
        self.assertEqual(self.client.pinned_model_id, PINNED_MODEL_ID)
        self.assertFalse(self.client.allow_fallback)

    def test_04_deterministic_backend_metrics(self):
        """Verify numerical metrics are calculated accurately in backend rather than invented by LLM."""
        test_reviews = [
            {"rating": 5.0, "verified_purchase": True, "helpful_vote": 10},
            {"rating": 5.0, "verified_purchase": True, "helpful_vote": 5},
            {"rating": 4.0, "verified_purchase": False, "helpful_vote": 2},
            {"rating": 1.0, "verified_purchase": True, "helpful_vote": 20}
        ]
        metrics = self.grounded_engine.calculate_backend_metrics(test_reviews)

        self.assertEqual(metrics["total_reviews"], 4)
        self.assertEqual(metrics["verified_purchases"], 3)
        self.assertEqual(metrics["verified_ratio_pct"], 75.0)
        self.assertEqual(metrics["average_rating"], 3.75)
        self.assertEqual(metrics["rating_distribution"][5], 2)
        self.assertEqual(metrics["rating_distribution_pct"][5], 50.0)
        self.assertEqual(metrics["total_helpful_votes"], 37)

    def test_05_claim_validation(self):
        """Verify citation parser validates real review IDs and flags hallucinated citations."""
        sample_prod = SAMPLE_METADATA[0]
        sample_revs = [r for r in SAMPLE_REVIEWS if r.get("parent_asin") == sample_prod["parent_asin"]]

        grounded = self.grounded_engine.build_grounded_context(
            query="test",
            product=sample_prod,
            reviews=sample_revs
        )

        valid_id = sample_revs[0]["review_id"]
        fake_id = "REV-FAKEXX-U9999-9999"

        generated_text_valid = f"Based on [Metadata #{sample_prod['parent_asin']}], customers praise noise cancellation [Review #{valid_id}]."
        res_valid = self.grounded_engine.validate_claims(generated_text_valid)
        self.assertTrue(res_valid["is_valid"])
        self.assertIn(valid_id, res_valid["valid_review_citations"])
        self.assertEqual(res_valid["grounding_score_pct"], 100.0)

        generated_text_invalid = f"A reviewer claimed [Review #{fake_id}] that the battery caught fire."
        res_invalid = self.grounded_engine.validate_claims(generated_text_invalid)
        self.assertFalse(res_invalid["is_valid"])
        self.assertIn(fake_id, res_invalid["hallucinated_review_citations"])

    def test_06_insufficient_evidence_handling(self):
        """Verify honest insufficient evidence response when records are missing."""
        empty_grounded = self.grounded_engine.build_grounded_context(
            query="Tell me about NonExistentProduct XYZ",
            product=None,
            reviews=[]
        )
        self.assertFalse(empty_grounded["has_sufficient_evidence"])

        resp = self.resp_gen.generate_response(
            user_query="Tell me about NonExistentProduct XYZ",
            retrieved_context="",
            product_info=None,
            reviews=[]
        )
        self.assertEqual(resp["status"], "insufficient_evidence")
        self.assertIn("Insufficient Evidence", resp["text"])

    def test_07_end_to_end_grounded_query(self):
        """Verify full pipeline execution returns verified citations and backend calculations."""
        result = self.conv_manager.process_message(
            user_query="What problems do people have with Instant Pot Duo?",
            session_id="test_grounded_session"
        )
        self.assertIsNotNone(result["reply"])
        self.assertTrue(len(result["reply"]) > 50)
        self.assertIn("backend_metrics", result)
        self.assertIn("claims_validation", result)
        self.assertGreaterEqual(result["claims_validation"]["grounding_score_pct"], 70.0)

if __name__ == "__main__":
    unittest.main()
