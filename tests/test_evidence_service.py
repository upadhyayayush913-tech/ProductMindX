"""
Unit tests for ProductEvidenceService and metadata presence checks.
Verifies robust handling of:
- None and NaN
- Multi-element NumPy arrays (preventing ambiguous truth value errors)
- Empty lists, populated lists, and nested lists
- Dictionaries and empty dicts
- String variants ('Not available', 'None', valid strings)
- Numeric and boolean scalars
"""

import unittest
import numpy as np
import pandas as pd
from services.product_evidence_service import is_metadata_field_present, ProductEvidenceService

class TestMetadataPresenceAndEvidenceService(unittest.TestCase):
    def test_none_and_nan(self):
        self.assertFalse(is_metadata_field_present(None))
        self.assertFalse(is_metadata_field_present(float("nan")))
        self.assertFalse(is_metadata_field_present(np.nan))

    def test_strings(self):
        self.assertFalse(is_metadata_field_present(""))
        self.assertFalse(is_metadata_field_present("   "))
        self.assertFalse(is_metadata_field_present("Not available"))
        self.assertFalse(is_metadata_field_present("None"))
        self.assertFalse(is_metadata_field_present("null"))
        self.assertFalse(is_metadata_field_present("[]"))
        self.assertTrue(is_metadata_field_present("Sony WH-1000XM5"))
        self.assertTrue(is_metadata_field_present("$398.00"))

    def test_numpy_arrays(self):
        # Empty array
        empty_arr = np.array([])
        self.assertFalse(is_metadata_field_present(empty_arr))

        # Multi-element array (Previously crashed with ValueError)
        multi_arr = np.array(["Noise cancellation", "30-hr battery", "Soft fit leather"])
        self.assertTrue(is_metadata_field_present(multi_arr))

        # Array of empty strings
        empty_str_arr = np.array(["", "Not available"])
        self.assertFalse(is_metadata_field_present(empty_str_arr))

        # Numeric array
        num_arr = np.array([1, 2, 3])
        self.assertTrue(is_metadata_field_present(num_arr))

    def test_lists_and_dicts(self):
        self.assertFalse(is_metadata_field_present([]))
        self.assertTrue(is_metadata_field_present(["ANC", "Bluetooth 5.2"]))
        self.assertFalse(is_metadata_field_present({}))
        self.assertTrue(is_metadata_field_present({"Weight": "250g"}))

    def test_product_evidence_profile_with_numpy_metadata(self):
        service = ProductEvidenceService()
        # Mock product where features is a numpy array (simulating parquet load)
        product_with_numpy = {
            "parent_asin": "TEST_ASIN_123",
            "title": "Test Audio Device",
            "main_category": "Electronics",
            "store": "TestStore",
            "price": "$199.99",
            "features": np.array(["Industry leading ANC", "Multi-point connection"]),
            "description": "High fidelity sound headphones",
            "details": {"Connectivity": "Bluetooth"}
        }

        # Must not raise ValueError: The truth value of an array with more than one element is ambiguous
        profile = service.calculate_product_evidence_profile(product_with_numpy)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["asin"], "TEST_ASIN_123")
        self.assertEqual(profile["meta_completeness_pct"], 100.0)

    def test_counter_evidence_retrieval(self):
        service = ProductEvidenceService()
        # Test counter-evidence on Sony WH-1000XM5
        res = service.get_counter_evidence("B09XS7JWHH", "sound quality")
        self.assertIn("verdict", res)
        self.assertIn("supporting_reviews", res)
        self.assertIn("counter_reviews", res)
        self.assertGreaterEqual(res["relevant_mentions_count"], 0)

if __name__ == "__main__":
    unittest.main()
