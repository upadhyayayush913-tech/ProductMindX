"""
Unified Product Evidence Service for ProductMind X.
Single Source of Truth across:
- Product Catalog & Search
- Product DNA
- Best-Supported Products Discovery
- Head-to-Head Comparison Engine
- Ask ProductMind Conversational Interface
- Explore Insights & Issue Detection
- System & Evaluation Dashboard
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from config import CATEGORIES
from data.data_loader import AmazonDataLoader
from utils.helpers import timestamp_to_str
from utils.logger import get_logger

logger = get_logger("evidence_service")

def is_metadata_field_present(val: Any) -> bool:
    """
    Safely determine if a metadata field is present and non-empty.
    Explicitly handles:
    - None
    - float / NaN
    - numpy ndarray (avoiding ambiguous truth value)
    - list / tuple / set
    - dict
    - string
    - numbers / booleans / scalars
    """
    if val is None:
        return False

    # Check for pandas/numpy NaN
    try:
        if isinstance(val, (float, int)) and pd.isna(val):
            return False
    except Exception:
        pass

    # Check for sequence / array (numpy ndarray, list, tuple, set)
    if hasattr(val, "__len__") and not isinstance(val, (str, bytes, dict)):
        try:
            if len(val) == 0:
                return False
            # Check if at least one meaningful element is present
            return any(is_metadata_field_present(item) for item in val)
        except Exception:
            return False

    # Check for dictionary
    if isinstance(val, dict):
        return len(val) > 0

    # Check for string
    if isinstance(val, (str, bytes)):
        s = str(val).strip()
        if not s or s.lower() in ["", "not available", "none", "null", "n/a", "[]", "{}"]:
            return False
        return True

    # Other scalar values (numbers, booleans)
    return True

class ProductEvidenceService:
    def __init__(self, loader: Optional[AmazonDataLoader] = None):
        self.loader = loader or AmazonDataLoader()
        self._cache_products: Optional[List[Dict[str, Any]]] = None
        self._cache_reviews_by_asin: Dict[str, List[Dict[str, Any]]] = {}

    def refresh(self):
        """Invalidate in-memory caches."""
        self._cache_products = None
        self._cache_reviews_by_asin = {}

    def get_all_products(self, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return all catalog products, optionally filtered by category."""
        if self._cache_products is None:
            self._cache_products = self.loader.get_all_products()

        if category_filter:
            return [p for p in self._cache_products if p.get("main_category") == category_filter]
        return list(self._cache_products)

    def get_product_by_id(self, asin: str) -> Optional[Dict[str, Any]]:
        """Resolve product by parent_asin or asin."""
        products = self.get_all_products()
        for p in products:
            if (p.get("parent_asin") or p.get("asin")) == asin:
                return p
        return None

    def get_product_reviews(self, asin: str) -> List[Dict[str, Any]]:
        """Return all indexed reviews for a specific product ID."""
        if asin in self._cache_reviews_by_asin:
            return self._cache_reviews_by_asin[asin]

        product = self.get_product_by_id(asin)
        if not product:
            return []

        cat = product.get("main_category", "Electronics")
        try:
            rev_df = self.loader.load_reviews(cat)
            matched = rev_df[rev_df["parent_asin"] == asin]
            revs = matched.to_dict("records")
        except Exception as e:
            logger.warning(f"Error loading reviews for {asin}: {e}")
            revs = []

        self._cache_reviews_by_asin[asin] = revs
        return revs

    def calculate_product_evidence_profile(self, product: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate complete evidence metrics for a product:
        - Total reviews indexed
        - Usable text count (non-empty review body)
        - Valid ratings count
        - Usable timestamps count
        - Verified-purchase count
        - Metadata completeness score
        - Rating distribution
        - Temporal observation range
        - Eligibility flags for analysis, comparison, and trend analysis
        """
        asin = product.get("parent_asin") or product.get("asin")
        reviews = self.get_product_reviews(asin)

        total_revs = len(reviews)
        usable_text = sum(1 for r in reviews if len(str(r.get("text", "")).strip()) >= 15)
        valid_ratings = sum(1 for r in reviews if r.get("rating") is not None)
        usable_timestamps = sum(1 for r in reviews if r.get("timestamp") and int(r.get("timestamp", 0)) > 0)
        verified_count = sum(1 for r in reviews if r.get("verified_purchase", True))

        # Metadata completeness
        meta_fields = ["title", "store", "price", "features", "description", "details"]
        present_fields = sum(1 for f in meta_fields if is_metadata_field_present(product.get(f)))
        meta_completeness_pct = round((present_fields / len(meta_fields)) * 100, 1)

        # Star breakdown
        star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in reviews:
            rating = r.get("rating")
            if rating is not None:
                star = max(1, min(5, int(round(float(rating)))))
                star_counts[star] += 1

        distinct_stars = sum(1 for count in star_counts.values() if count > 0)

        # Dates
        timestamps = [int(r["timestamp"]) for r in reviews if r.get("timestamp") and int(r["timestamp"]) > 0]
        start_date = timestamp_to_str(min(timestamps)) if timestamps else "N/A"
        end_date = timestamp_to_str(max(timestamps)) if timestamps else "N/A"

        # Task Eligibility
        eligible_for_basic_analysis = total_revs >= 1 and usable_text >= 1
        eligible_for_comparison = total_revs >= 2 and usable_text >= 2
        eligible_for_trend = total_revs >= 3 and usable_timestamps >= 3 and distinct_stars >= 2

        # Evidence Strength Score (0 to 100)
        # Weighted: Review count (40%), Usable text ratio (25%), Timestamps (15%), Meta completeness (20%)
        text_ratio = (usable_text / total_revs) if total_revs > 0 else 0.0
        ts_ratio = (usable_timestamps / total_revs) if total_revs > 0 else 0.0
        vol_score = min(40.0, total_revs * 4.0) # 10+ reviews gives full 40 pts

        evidence_score = round(
            vol_score +
            (text_ratio * 25.0) +
            (ts_ratio * 15.0) +
            (meta_completeness_pct * 0.20),
            1
        )

        return {
            "asin": asin,
            "title": product.get("title", ""),
            "category": product.get("main_category", ""),
            "store": product.get("store", ""),
            "price": product.get("price", "Not available"),
            "total_reviews": total_revs,
            "usable_text_count": usable_text,
            "valid_ratings_count": valid_ratings,
            "usable_timestamps_count": usable_timestamps,
            "verified_purchase_count": verified_count,
            "verified_ratio_pct": round((verified_count / total_revs * 100), 1) if total_revs > 0 else 0.0,
            "meta_completeness_pct": meta_completeness_pct,
            "star_distribution": star_counts,
            "distinct_rating_tiers": distinct_stars,
            "observation_window": {"start": start_date, "end": end_date},
            "eligibility": {
                "basic_analysis": eligible_for_basic_analysis,
                "comparison": eligible_for_comparison,
                "trend_analysis": eligible_for_trend
            },
            "evidence_score": evidence_score
        }

    def get_best_supported_products(
        self,
        category: Optional[str] = None,
        min_reviews: int = 3,
        task: str = "general" # 'general', 'comparison', 'trend'
    ) -> List[Dict[str, Any]]:
        """
        Identify and rank products with the strongest actual evidence coverage.
        Discovers products dynamically based on locally indexed review density.
        """
        products = self.get_all_products(category_filter=category)
        evaluated = []

        for p in products:
            prof = self.calculate_product_evidence_profile(p)
            if prof["total_reviews"] < min_reviews:
                continue

            if task == "trend" and not prof["eligibility"]["trend_analysis"]:
                continue
            if task == "comparison" and not prof["eligibility"]["comparison"]:
                continue

            # Qualification Reason
            reasons = [
                f"{prof['total_reviews']} indexed reviews ({prof['verified_ratio_pct']}% verified)",
                f"{prof['usable_text_count']} detailed review passages",
                f"Active observation window: {prof['observation_window']['start']} to {prof['observation_window']['end']}"
            ]
            if prof["distinct_rating_tiers"] >= 3:
                reasons.append("Multi-tier star distribution (diverse customer opinions)")

            prof["qualification_reason"] = " • ".join(reasons)
            evaluated.append(prof)

        # Rank by evidence score descending
        evaluated.sort(key=lambda x: x["evidence_score"], reverse=True)
        return evaluated

    def get_corpus_evidence_summary(self) -> Dict[str, Any]:
        """Unified corpus metrics for dashboard and evaluation."""
        products = self.get_all_products()
        total_prods = len(products)
        total_revs = 0
        usable_text_total = 0
        usable_ts_total = 0
        verified_total = 0

        categories_summary = defaultdict(lambda: {"products": 0, "reviews": 0})

        for p in products:
            prof = self.calculate_product_evidence_profile(p)
            cat = prof["category"]
            categories_summary[cat]["products"] += 1
            categories_summary[cat]["reviews"] += prof["total_reviews"]

            total_revs += prof["total_reviews"]
            usable_text_total += prof["usable_text_count"]
            usable_ts_total += prof["usable_timestamps_count"]
            verified_total += prof["verified_purchase_count"]

        return {
            "total_products": total_prods,
            "total_reviews": total_revs,
            "usable_text_total": usable_text_total,
            "usable_timestamps_total": usable_ts_total,
            "verified_purchases_total": verified_total,
            "verified_ratio_pct": round((verified_total / total_revs * 100), 1) if total_revs > 0 else 0.0,
            "categories_summary": dict(categories_summary),
            "configured_categories": CATEGORIES
        }

    def get_counter_evidence(
        self,
        asin: str,
        topic_or_aspect: str
    ) -> Dict[str, Any]:
        """
        Systematically retrieve supporting and contradictory evidence for a specific product and issue/aspect.
        Prevents confirmation bias by actively surfacing positive counter-evidence alongside negative complaints.
        """
        product = self.get_product_by_id(asin)
        reviews = self.get_product_reviews(asin)

        if not product or not reviews:
            return {
                "asin": asin,
                "topic": topic_or_aspect,
                "status": "no_data",
                "supporting_count": 0,
                "counter_count": 0,
                "supporting_reviews": [],
                "counter_reviews": [],
                "verdict": "Insufficient data recorded for this product."
            }

        topic_clean = topic_or_aspect.lower().strip()
        # Derive keyword tokens
        keywords = [k.strip() for k in topic_clean.replace("/", " ").replace("-", " ").split() if len(k.strip()) >= 3]
        if not keywords:
            keywords = [topic_clean]

        supporting = [] # Reviews reporting the issue (critical rating or negative sentiment mentioning topic)
        counter = []    # Reviews reporting zero problems or high satisfaction with this aspect

        for r in reviews:
            text = f"{r.get('title', '')} {r.get('text', '')}".lower()
            rating = float(r.get("rating", 3.0))
            rev_id = r.get("review_id", "N/A")
            vp = r.get("verified_purchase", True)
            ts = r.get("timestamp")

            has_topic_mention = any(kw in text for kw in keywords)

            if has_topic_mention:
                if rating <= 3.0:
                    supporting.append({
                        "review_id": rev_id,
                        "rating": rating,
                        "verified_purchase": vp,
                        "timestamp": ts,
                        "headline": r.get("title", ""),
                        "text": r.get("text", ""),
                        "stance": "Supporting Complaint"
                    })
                elif rating >= 4.0:
                    counter.append({
                        "review_id": rev_id,
                        "rating": rating,
                        "verified_purchase": vp,
                        "timestamp": ts,
                        "headline": r.get("title", ""),
                        "text": r.get("text", ""),
                        "stance": "Counter-Evidence (Favorable Experience)"
                    })
            else:
                # If rating is 5.0 and no issue mentioned, it constitutes general counter-evidence
                if rating >= 4.5 and len(counter) < 3:
                    counter.append({
                        "review_id": rev_id,
                        "rating": rating,
                        "verified_purchase": vp,
                        "timestamp": ts,
                        "headline": r.get("title", ""),
                        "text": r.get("text", ""),
                        "stance": "Counter-Evidence (Zero Defects Reported)"
                    })

        total_relevant = len(supporting) + len(counter)
        support_pct = round((len(supporting) / total_relevant * 100), 1) if total_relevant > 0 else 0.0
        counter_pct = round((len(counter) / total_relevant * 100), 1) if total_relevant > 0 else 0.0

        if len(supporting) == 0 and len(counter) == 0:
            verdict = f"No specific customer mentions found regarding '{topic_or_aspect}'."
        elif len(supporting) > 0 and len(counter) == 0:
            verdict = f"Unanimous complaints: All {len(supporting)} relevant reviews reported issues with '{topic_or_aspect}'."
        elif len(supporting) == 0 and len(counter) > 0:
            verdict = f"Unanimous satisfaction: All {len(counter)} relevant reviews reported positive experiences with '{topic_or_aspect}'."
        else:
            verdict = (
                f"Nuanced evidence: {len(supporting)} reviewers ({support_pct}%) reported issues, "
                f"while {len(counter)} reviewers ({counter_pct}%) reported reliable/satisfactory performance."
            )

        return {
            "asin": asin,
            "product_title": product.get("title", ""),
            "topic": topic_or_aspect,
            "total_product_reviews": len(reviews),
            "relevant_mentions_count": total_relevant,
            "supporting_count": len(supporting),
            "counter_count": len(counter),
            "support_ratio_pct": support_pct,
            "counter_ratio_pct": counter_pct,
            "verdict": verdict,
            "supporting_reviews": supporting,
            "counter_reviews": counter
        }
