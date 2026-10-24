"""
Unit tests for Dataset Coverage Ledger, Ingestion Reconciliation, and Sensitivity Analysis.
Verifies:
1. Manifest generation and correct distinction between published and locally indexed counts.
2. Ingestion pipeline stages and data integrity.
3. Stable review IDs and zero duplicates across parquet files.
4. Preference-weighted recommendation sensitivity analysis.
"""

import unittest
from pathlib import Path
import json

from config import CACHE_DIR, CATEGORIES
from data.coverage_ledger import coverage_ledger, PUBLISHED_CORPUS_BENCHMARKS
from data.data_loader import AmazonDataLoader
from analytics.tools import (
    calculate_preference_weighted_recommendation,
    calculate_weight_sensitivity_analysis
)
from services.product_evidence_service import ProductEvidenceService

class TestCoverageAndIngestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = coverage_ledger
        cls.manifest = cls.ledger.audit_local_storage()
        cls.loader = AmazonDataLoader()
        cls.evidence_service = ProductEvidenceService(loader=cls.loader)

    def test_01_manifest_generation(self):
        """Verify coverage manifest is created and valid JSON on disk."""
        manifest_path = CACHE_DIR / "dataset_coverage_manifest.json"
        self.assertTrue(manifest_path.exists())
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["manifest_version"], "2.0-research")
        self.assertIn("pipeline_summary", data)
        self.assertIn("processing_stage_ledger", data)

    def test_02_published_vs_local_count_distinction(self):
        """Verify published corpus counts are strictly documented and never confused with local storage."""
        summary = self.manifest["pipeline_summary"]
        published_reviews = summary["published_source_reviews_documented"]
        local_reviews = summary["total_reviews_stored"]

        # Published size is > 100M
        self.assertGreater(published_reviews, 100_000_000)
        # Local size is a curated, verified subset
        self.assertLess(local_reviews, 100_000)
        self.assertNotEqual(published_reviews, local_reviews)

    def test_03_zero_duplicate_review_ids(self):
        """Verify all cached parquet reviews have unique, non-null review IDs."""
        total_rows = 0
        all_ids = set()
        for cat in CATEGORIES:
            df = self.loader.load_reviews(cat)
            total_rows += len(df)
            self.assertEqual(df["review_id"].isna().sum(), 0, f"Found null review_ids in {cat}")
            cat_ids = set(df["review_id"].tolist())
            self.assertEqual(len(df), len(cat_ids), f"Duplicate review_ids inside {cat}")
            all_ids.update(cat_ids)
        self.assertEqual(total_rows, len(all_ids), "Duplicate review_ids found across categories")

    def test_04_processing_stages_reconciled(self):
        """Verify all 5 processing stages are populated with metrics."""
        stages = self.manifest["processing_stage_ledger"]
        self.assertIn("stage_1_source_available", stages)
        self.assertIn("stage_2_locally_stored_parquet", stages)
        self.assertIn("stage_3_absa_nlp_analyzed", stages)
        self.assertIn("stage_4_vector_indexed_chromadb", stages)
        self.assertIn("stage_5_lexical_bm25_search", stages)

    def test_05_weight_sensitivity_analysis(self):
        """Verify sensitivity analysis computes multi-scenario trade-off stability."""
        products = self.evidence_service.get_all_products()
        prods_to_compare = products[:2]
        asin1 = prods_to_compare[0].get("parent_asin") or prods_to_compare[0].get("asin")
        asin2 = prods_to_compare[1].get("parent_asin") or prods_to_compare[1].get("asin")

        metrics_map = {
            asin1: {"total_reviews": 10, "average_rating": 4.6, "verified_ratio_pct": 100.0},
            asin2: {"total_reviews": 8, "average_rating": 4.4, "verified_ratio_pct": 100.0}
        }
        insights_map = {
            asin1: {
                "pros": [{"aspect": "sound quality", "positive_percentage": 95, "mention_count": 5}],
                "cons": [{"aspect": "comfort", "positive_percentage": 40, "mention_count": 2}],
                "issues": []
            },
            asin2: {
                "pros": [{"aspect": "comfort", "positive_percentage": 90, "mention_count": 4}],
                "cons": [{"aspect": "sound quality", "positive_percentage": 50, "mention_count": 2}],
                "issues": []
            }
        }

        sens = calculate_weight_sensitivity_analysis(
            candidate_products=prods_to_compare,
            metrics_by_asin=metrics_map,
            insights_by_asin=insights_map
        )
        self.assertIn("dominant_winner", sens)
        self.assertIn("stability_pct", sens)
        self.assertEqual(len(sens["scenarios"]), 5)
        self.assertIn("Balanced Default", sens["scenarios"])
        self.assertIn("Acoustic / Audiophile", sens["scenarios"])

if __name__ == "__main__":
    unittest.main()
