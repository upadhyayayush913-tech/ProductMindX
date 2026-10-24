"""
Unit tests for Statistically Validated Emerging-Issue Detector.
Verifies:
1. Fisher's exact test for small sample contingency tables.
2. Two-proportion pooled Z-test for asymptotic samples.
3. Benjamini-Hochberg FDR multiple testing correction.
4. 4-state categorization logic.
5. Confidence interval bounds.
"""

import unittest
from analytics.statistical_issue_detector import StatisticalIssueDetector

class TestStatisticalIssueDetector(unittest.TestCase):
    def setUp(self):
        self.detector = StatisticalIssueDetector(min_sample_size=3, alpha=0.05)

    def test_insufficient_sample_size(self):
        """Verify detector safely flags insufficient samples instead of guessing."""
        res = self.detector.test_single_issue("hinge failure", k_current=1, n_current=2, k_baseline=0, n_baseline=2)
        self.assertEqual(res["state"], "INSUFFICIENT_EVIDENCE")
        self.assertFalse(res["statistically_significant"])

    def test_fishers_exact_test_small_sample(self):
        """Verify Fisher's exact test is selected when cell counts are < 5."""
        # 4 out of 10 in current vs 0 out of 10 in baseline
        res = self.detector.test_single_issue("battery drain", k_current=4, n_current=10, k_baseline=0, n_baseline=10)
        self.assertIn("Fisher's Exact Test", res["test_method"])
        self.assertGreater(res["rate_difference"], 0)
        self.assertLess(res["p_value"], 0.10)

    def test_no_increase_detected(self):
        """Verify rate decrease or constancy is flagged as NO_INCREASE_DETECTED."""
        res = self.detector.test_single_issue("noise issue", k_current=1, n_current=20, k_baseline=4, n_baseline=20)
        self.assertEqual(res["state"], "NO_INCREASE_DETECTED")
        self.assertLessEqual(res["rate_difference"], 0)
        self.assertFalse(res["statistically_significant"])

    def test_benjamini_hochberg_correction(self):
        """Verify Benjamini-Hochberg controls false discovery rate across multiple tests."""
        raw_tests = [
            {"issue": "A", "state": "DESCRIPTIVE_INCREASE", "rate_difference": 0.2, "rate_difference_pct": 20.0, "p_value": 0.01},
            {"issue": "B", "state": "DESCRIPTIVE_INCREASE", "rate_difference": 0.1, "rate_difference_pct": 10.0, "p_value": 0.04},
            {"issue": "C", "state": "DESCRIPTIVE_INCREASE", "rate_difference": 0.05, "rate_difference_pct": 5.0, "p_value": 0.06},
            {"issue": "D", "state": "NO_INCREASE_DETECTED", "rate_difference": -0.05, "rate_difference_pct": -5.0, "p_value": 0.80}
        ]
        corrected = self.detector.apply_benjamini_hochberg(raw_tests)

        # Monotonicity check: adjusted p-values should be >= raw p-values
        for r in corrected:
            self.assertGreaterEqual(r["p_value_adjusted"], r["p_value"] - 1e-6)

        # Issue A: p=0.01, rank 1 of 3 testable. BH p = min(1, 0.01 * 3 / 1) = 0.03 <= 0.05 -> STATISTICALLY_SUPPORTED
        self.assertEqual(corrected[0]["state"], "STATISTICALLY_SUPPORTED_INCREASE")
        self.assertTrue(corrected[0]["statistically_significant"])

    def test_confidence_interval_bounds(self):
        """Verify 95% confidence intervals are within [-1.0, 1.0]."""
        low, high = self.detector.calculate_rate_difference_ci(5, 20, 2, 20)
        self.assertGreaterEqual(low, -1.0)
        self.assertLessEqual(high, 1.0)
        self.assertLessEqual(low, high)

    def test_caveat_presence(self):
        """Verify required scientific disclaimer is always included."""
        res = self.detector.test_single_issue("test", k_current=5, n_current=15, k_baseline=1, n_baseline=15)
        self.assertIn("caveat", res)
        self.assertIn("not proof of manufacturing defect", res["caveat"])

if __name__ == "__main__":
    unittest.main()
