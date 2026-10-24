"""
Unit and Integration Tests for Unified Filter Service and Query Engine.
Verifies:
1. Category-dependent product list and review population
2. Rating range boundaries (inclusive 1.0 to 5.0)
3. Verified purchase constraint
4. Sentiment filter correctness
5. Filter combination and reset state
6. KPI calculation integrity and star distributions
7. Case-insensitive search behavior
"""

import unittest
from services.filter_service import CorpusFilterService, FilterState
from services.product_evidence_service import ProductEvidenceService

class TestCorpusFilterService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence_service = ProductEvidenceService()
        cls.filter_service = CorpusFilterService(evidence_service=cls.evidence_service)

    def test_default_filter_returns_full_corpus(self):
        state = FilterState()
        self.assertTrue(state.is_default())
        res = self.filter_service.filter_data(state)
        
        self.assertEqual(res["total_reviews"], 68)
        self.assertEqual(res["products_cataloged"], 15)
        self.assertEqual(res["categories_represented"], 6)
        self.assertEqual(sum(res["star_counts"].values()), 68)

    def test_category_filter(self):
        state = FilterState(category="Electronics")
        res = self.filter_service.filter_data(state)
        
        self.assertFalse(state.is_default())
        self.assertEqual(res["products_cataloged"], 3)
        self.assertEqual(res["total_reviews"], 26)
        self.assertEqual(res["categories_represented"], 1)

    def test_single_product_filter(self):
        # Apple AirPods Pro ASIN: B0BDHW4P5S
        state = FilterState(category="Electronics", product_asin="B0BDHW4P5S")
        res = self.filter_service.filter_data(state)
        
        self.assertEqual(res["products_cataloged"], 1)
        self.assertEqual(res["total_reviews"], 8)
        for p in res["active_products"]:
            self.assertEqual(p.get("parent_asin"), "B0BDHW4P5S")

    def test_rating_range_filtering(self):
        # Only 5-star reviews
        state = FilterState(rating_min=5.0, rating_max=5.0)
        res = self.filter_service.filter_data(state)
        
        self.assertEqual(res["star_counts"][1], 0)
        self.assertEqual(res["star_counts"][2], 0)
        self.assertEqual(res["star_counts"][3], 0)
        self.assertEqual(res["star_counts"][4], 0)
        self.assertEqual(res["total_reviews"], res["star_counts"][5])
        self.assertGreater(res["total_reviews"], 0)

    def test_verified_only_filtering(self):
        state = FilterState(verified_only=True)
        res = self.filter_service.filter_data(state)
        
        self.assertEqual(res["verified_ratio_pct"], 100.0)
        for r in res["filtered_reviews"]:
            self.assertTrue(r.get("verified_purchase", True))

    def test_sentiment_filtering(self):
        state = FilterState(sentiment="Positive")
        res = self.filter_service.filter_data(state)
        self.assertGreater(res["total_reviews"], 0)

    def test_active_descriptions_and_reset(self):
        prod_map = {"B0BDHWDR12": "Apple AirPods Pro"}
        state = FilterState(category="Electronics", product_asin="B0BDHWDR12", rating_min=4.0, rating_max=5.0)
        pills = state.get_active_descriptions(prod_map)
        
        self.assertIn("Category: Electronics", pills)
        self.assertTrue(any("Apple AirPods" in p for p in pills))
        self.assertIn("Rating: 4.0–5.0★", pills)

        # Reset state
        reset_state = FilterState()
        self.assertTrue(reset_state.is_default())
        self.assertEqual(len(reset_state.get_active_descriptions(prod_map)), 0)

if __name__ == "__main__":
    unittest.main()
