r"""
Statistically Validated Emerging-Issue and Complaint Velocity Detector for ProductMind X.

Implements rigorous hypothesis testing and epidemiological surveillance for consumer product complaints:
1. Explicit Observation Periods: Current vs Baseline temporal windows.
2. Transparent Denominators: Total product reviews in each period ($N_1, N_2$).
3. Two-Proportion Hypothesis Testing:
   - Fisher's Exact Test (for small cell counts < 5)
   - Pooled Two-Proportion Z-Test (for asymptotic samples)
4. Benjamini-Hochberg (BH) Multiple Testing Correction (FDR control across multiple concurrent issues).
5. 95% Confidence Intervals for Rate Differences ($\Delta p$).
6. Rigorous 4-State Evidence Categorization:
   - INSUFFICIENT_EVIDENCE
   - DESCRIPTIVE_INCREASE
   - STATISTICALLY_SUPPORTED_INCREASE
   - NO_INCREASE_DETECTED
7. Explicit Scientific Guardrails: Disclaims defect causation or manufacturing fault proof.
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import math
import numpy as np
from scipy import stats

from utils.helpers import timestamp_to_str
from utils.logger import get_logger

logger = get_logger("statistical_issue_detector")

DEFAULT_MIN_SAMPLE_SIZE = 3
DEFAULT_ALPHA = 0.05

class StatisticalIssueDetector:
    def __init__(
        self,
        min_sample_size: int = DEFAULT_MIN_SAMPLE_SIZE,
        alpha: float = DEFAULT_ALPHA
    ):
        self.min_sample_size = min_sample_size
        self.alpha = alpha

    def partition_temporal_windows(
        self,
        reviews: List[Dict[str, Any]],
        split_ratio: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Partition reviews chronologically into baseline (older) and current (recent) windows.
        If timestamps are missing, uses review list indexing order.
        """
        valid_ts_reviews = []
        for r in reviews:
            ts = r.get("timestamp")
            if ts and int(ts) > 0:
                valid_ts_reviews.append(r)

        if len(valid_ts_reviews) >= self.min_sample_size * 2:
            sorted_revs = sorted(valid_ts_reviews, key=lambda x: int(x["timestamp"]))
        else:
            sorted_revs = list(reviews)

        n = len(sorted_revs)
        if n < self.min_sample_size * 2:
            return [], [], {"error": f"Total samples ({n}) insufficient for 2-period temporal partition"}

        split_idx = int(n * (1.0 - split_ratio))
        baseline_revs = sorted_revs[:split_idx]
        current_revs = sorted_revs[split_idx:]

        # Extract date ranges
        def get_date_bounds(rev_subset):
            ts_list = [int(r["timestamp"]) for r in rev_subset if r.get("timestamp") and int(r["timestamp"]) > 0]
            if not ts_list:
                return "Unknown", "Unknown"
            min_ts, max_ts = min(ts_list), max(ts_list)
            # handle ms vs s
            if min_ts > 1e11:
                return datetime.fromtimestamp(min_ts / 1000).strftime("%Y-%m-%d"), datetime.fromtimestamp(max_ts / 1000).strftime("%Y-%m-%d")
            return datetime.fromtimestamp(min_ts).strftime("%Y-%m-%d"), datetime.fromtimestamp(max_ts).strftime("%Y-%m-%d")

        b_start, b_end = get_date_bounds(baseline_revs)
        c_start, c_end = get_date_bounds(current_revs)

        window_meta = {
            "baseline_window": {"start": b_start, "end": b_end, "sample_size": len(baseline_revs)},
            "current_window": {"start": c_start, "end": c_end, "sample_size": len(current_revs)}
        }

        return baseline_revs, current_revs, window_meta

    def calculate_rate_difference_ci(
        self,
        k1: int, N1: int,
        k2: int, N2: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute 95% Confidence Interval for proportion difference (p1 - p2)
        using Agresti-Caffo adjustment for small samples or standard Wald.
        """
        if N1 <= 0 or N2 <= 0:
            return 0.0, 0.0

        p1 = k1 / N1
        p2 = k2 / N2
        diff = p1 - p2

        z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
        # Agresti-Caffo adjustment: add 1 success and 2 observations to each group
        p1_adj = (k1 + 1) / (N1 + 2)
        p2_adj = (k2 + 1) / (N2 + 2)
        se_adj = math.sqrt((p1_adj * (1 - p1_adj) / (N1 + 2)) + (p2_adj * (1 - p2_adj) / (N2 + 2)))

        ci_lower = max(-1.0, diff - z * se_adj)
        ci_upper = min(1.0, diff + z * se_adj)

        return round(ci_lower, 4), round(ci_upper, 4)

    def test_single_issue(
        self,
        issue_name: str,
        k_current: int,
        n_current: int,
        k_baseline: int,
        n_baseline: int
    ) -> Dict[str, Any]:
        """
        Perform statistical test comparing complaint proportion in current vs baseline period.
        Selects Fisher's exact test when any cell < 5, else pooled two-proportion z-test.
        """
        # 1. Check sample sizes
        if n_current < self.min_sample_size or n_baseline < self.min_sample_size:
            return {
                "issue": issue_name,
                "state": "INSUFFICIENT_EVIDENCE",
                "verdict": f"Sample size insufficient (Current: {n_current}, Baseline: {n_baseline}, Min required: {self.min_sample_size})",
                "baseline_count": k_baseline,
                "baseline_total": n_baseline,
                "baseline_rate": round(k_baseline / n_baseline, 4) if n_baseline > 0 else 0.0,
                "current_count": k_current,
                "current_total": n_current,
                "current_rate": round(k_current / n_current, 4) if n_current > 0 else 0.0,
                "rate_difference": 0.0,
                "relative_increase_pct": 0.0,
                "test_method": "None (Insufficient Sample)",
                "p_value": 1.0,
                "p_value_adjusted": 1.0,
                "statistically_significant": False,
                "ci_95": (0.0, 0.0)
            }

        p_curr = k_current / n_current
        p_base = k_baseline / n_baseline
        rate_diff = p_curr - p_base

        if p_base > 0:
            rel_inc_pct = round(((p_curr - p_base) / p_base) * 100, 1)
        else:
            rel_inc_pct = 100.0 if k_current > 0 else 0.0

        ci_low, ci_high = self.calculate_rate_difference_ci(k_current, n_current, k_baseline, n_baseline)

        # Build 2x2 contingency table:
        # [ [k_curr, n_curr - k_curr],
        #   [k_base, n_base - k_base] ]
        table = [
            [k_current, max(0, n_current - k_current)],
            [k_baseline, max(0, n_baseline - k_baseline)]
        ]

        use_fisher = any(cell < 5 for row in table for cell in row)

        if use_fisher:
            test_method = "Fisher's Exact Test (One-Sided Greater)"
            # alternative='greater': odds ratio > 1 (current rate > baseline rate)
            _, p_value = stats.fisher_exact(table, alternative="greater")
        else:
            test_method = "Two-Proportion Pooled Z-Test (One-Sided)"
            p_pool = (k_current + k_baseline) / (n_current + n_baseline)
            if p_pool == 0.0 or p_pool == 1.0:
                p_value = 1.0
            else:
                se = math.sqrt(p_pool * (1.0 - p_pool) * ((1.0 / n_current) + (1.0 / n_baseline)))
                z = (p_curr - p_base) / se
                p_value = 1.0 - stats.norm.cdf(z)

        p_value = float(np.clip(p_value, 0.0, 1.0))

        # Initial state determination (before FDR multiple-testing correction)
        if rate_diff <= 0:
            state = "NO_INCREASE_DETECTED"
            verdict = f"Complaint rate is stable or decreased ({round(p_curr*100, 1)}% current vs {round(p_base*100, 1)}% baseline)."
        elif p_value <= self.alpha:
            state = "STATISTICALLY_SUPPORTED_INCREASE"
            verdict = f"Statistically supported increase: complaint rate rose by +{round(rate_diff*100, 1)}% (p = {p_value:.4f} <= {self.alpha})."
        else:
            state = "DESCRIPTIVE_INCREASE"
            verdict = f"Descriptive increase observed (+{round(rate_diff*100, 1)}%), but not statistically significant at alpha={self.alpha} (p = {p_value:.4f})."

        return {
            "issue": issue_name,
            "state": state,
            "verdict": verdict,
            "baseline_count": k_baseline,
            "baseline_total": n_baseline,
            "baseline_rate": round(p_base, 4),
            "baseline_rate_pct": round(p_base * 100, 1),
            "current_count": k_current,
            "current_total": n_current,
            "current_rate": round(p_curr, 4),
            "current_rate_pct": round(p_curr * 100, 1),
            "rate_difference": round(rate_diff, 4),
            "rate_difference_pct": round(rate_diff * 100, 1),
            "relative_increase_pct": rel_inc_pct,
            "test_method": test_method,
            "p_value": round(p_value, 5),
            "p_value_adjusted": round(p_value, 5), # Will be updated by Benjamini-Hochberg
            "statistically_significant": (p_value <= self.alpha and rate_diff > 0),
            "ci_95": (ci_low, ci_high),
            "caveat": "Note: Statistical significance identifies a measurable spike in reported consumer reviews, not proof of manufacturing defect or physical causality."
        }

    def apply_benjamini_hochberg(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply Benjamini-Hochberg FDR correction to control false discovery rate
        across multiple simultaneous issue tests.
        """
        if not results:
            return []

        # Filter out tests with insufficient evidence
        testable_indices = [i for i, r in enumerate(results) if r["state"] != "INSUFFICIENT_EVIDENCE"]
        m = len(testable_indices)
        if m == 0:
            return results

        # Sort testable results by raw p-value
        sorted_indices = sorted(testable_indices, key=lambda i: results[i]["p_value"])

        # Calculate BH adjusted p-values
        # p_adj = min(1, p_val * m / rank)
        cum_min = 1.0
        # Iterate backwards to ensure monotonicity
        adjusted_p_values = {}
        for rank_rev, idx in enumerate(reversed(sorted_indices)):
            rank = m - rank_rev
            p_val = results[idx]["p_value"]
            bh_val = min(1.0, (p_val * m) / rank)
            cum_min = min(cum_min, bh_val)
            adjusted_p_values[idx] = round(cum_min, 5)

        # Apply adjusted p-values and update states
        for idx in testable_indices:
            adj_p = adjusted_p_values[idx]
            results[idx]["p_value_adjusted"] = adj_p
            rate_diff = results[idx]["rate_difference"]

            if rate_diff <= 0:
                results[idx]["state"] = "NO_INCREASE_DETECTED"
                results[idx]["statistically_significant"] = False
            elif adj_p <= self.alpha:
                results[idx]["state"] = "STATISTICALLY_SUPPORTED_INCREASE"
                results[idx]["statistically_significant"] = True
                results[idx]["verdict"] = (
                    f"Statistically supported increase: complaint rate rose by +{results[idx]['rate_difference_pct']}% "
                    f"(BH adjusted p = {adj_p:.4f} <= {self.alpha})."
                )
            else:
                results[idx]["state"] = "DESCRIPTIVE_INCREASE"
                results[idx]["statistically_significant"] = False
                results[idx]["verdict"] = (
                    f"Descriptive increase observed (+{results[idx]['rate_difference_pct']}%), "
                    f"but not statistically significant after BH FDR correction (adj p = {adj_p:.4f} > {self.alpha})."
                )

        return results

    def analyze_product_issues(
        self,
        reviews: List[Dict[str, Any]],
        known_issues: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute full temporal issue surveillance on a product's reviews.
        Detects recurring complaint triggers, counts occurrences across baseline and current windows,
        and executes multiple-testing corrected statistical tests.
        """
        baseline_revs, current_revs, window_meta = self.partition_temporal_windows(reviews)

        if not baseline_revs or not current_revs:
            return {
                "status": "insufficient_samples",
                "message": window_meta.get("error", "Insufficient reviews for temporal analysis"),
                "issues": []
            }

        # Define candidate issues to test
        if known_issues:
            candidate_issues = list(known_issues)
        else:
            # Common complaint keywords
            candidate_issues = [
                "battery drain / charging failure",
                "crackling / static noise",
                "disconnection / bluetooth drop",
                "uncomfortable / ear pain",
                "hinge / build durability",
                "burn error / overheating",
                "coating peeling / wear"
            ]

        # Count mentions in each window
        def count_mentions(rev_subset, keywords):
            count = 0
            for r in rev_subset:
                text = f"{r.get('title', '')} {r.get('text', '')}".lower()
                rating = float(r.get("rating", 3.0))
                # Count if review is critical/negative and mentions keyword
                if rating <= 3.5 and any(kw in text for kw in keywords):
                    count += 1
            return count

        issue_keyword_map = {
            "battery drain / charging failure": ["battery", "charge", "charging", "drain", "dead"],
            "crackling / static noise": ["crackle", "crackling", "static", "buzz", "buzzing", "hiss"],
            "disconnection / bluetooth drop": ["disconnect", "drop", "bluetooth", "cut out", "unpair"],
            "uncomfortable / ear pain": ["uncomfortable", "pain", "tight", "hurt", "earpad", "headband"],
            "hinge / build durability": ["hinge", "break", "broke", "fragile", "plastic", "snap"],
            "burn error / overheating": ["burn", "overheat", "spark", "hot", "thermal"],
            "coating peeling / wear": ["peel", "peeling", "coating", "flake", "wear", "sticky"]
        }

        n_curr = len(current_revs)
        n_base = len(baseline_revs)

        tested_results = []
        for issue in candidate_issues:
            kws = issue_keyword_map.get(issue, [issue.lower()])
            k_curr = count_mentions(current_revs, kws)
            k_base = count_mentions(baseline_revs, kws)

            res = self.test_single_issue(
                issue_name=issue,
                k_current=k_curr,
                n_current=n_curr,
                k_baseline=k_base,
                n_baseline=n_base
            )
            tested_results.append(res)

        # Apply Benjamini-Hochberg FDR correction across all candidate issues
        corrected_results = self.apply_benjamini_hochberg(tested_results)

        # Tally states
        state_counts = {
            "STATISTICALLY_SUPPORTED_INCREASE": 0,
            "DESCRIPTIVE_INCREASE": 0,
            "NO_INCREASE_DETECTED": 0,
            "INSUFFICIENT_EVIDENCE": 0
        }
        for r in corrected_results:
            state_counts[r["state"]] = state_counts.get(r["state"], 0) + 1

        return {
            "status": "success",
            "observation_windows": window_meta,
            "alpha": self.alpha,
            "multiple_testing_correction": "Benjamini-Hochberg (FDR)",
            "summary_counts": state_counts,
            "issues": corrected_results
        }

# Global singleton
statistical_issue_detector = StatisticalIssueDetector()
