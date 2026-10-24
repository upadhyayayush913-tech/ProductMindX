"""
Unified Filter and Query Service for ProductMind X.
Single Source of Truth for:
- Category Scope Filtering
- Dynamic Product Selection
- Rating Range (1.0 - 5.0)
- Date Range Filtering
- Sentiment Filtering (Positive, Neutral, Negative, or All)
- Verified Purchase Filtering
- Deterministic KPI & Chart Computations
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import datetime
import re

from config import CATEGORIES
from services.product_evidence_service import ProductEvidenceService, is_metadata_field_present
from utils.logger import get_logger

logger = get_logger("filter_service")

@dataclass
class FilterState:
    category: Optional[str] = None          # None means "All Categories"
    product_asin: Optional[str] = None      # None means "All Products"
    rating_min: float = 1.0
    rating_max: float = 5.0
    date_start: Optional[str] = None        # e.g. "2023-01-01"
    date_end: Optional[str] = None          # e.g. "2023-12-31"
    sentiment: Optional[str] = None         # "Positive", "Neutral", "Negative" or None
    verified_only: bool = False

    def is_default(self) -> bool:
        return (
            self.category is None
            and self.product_asin is None
            and self.rating_min <= 1.0
            and self.rating_max >= 5.0
            and not self.sentiment
            and not self.verified_only
        )

    def get_active_descriptions(self, products_map: Dict[str, str]) -> List[str]:
        """Return human-readable pills of applied filters."""
        pills = []
        if self.category:
            pills.append(f"Category: {self.category.replace('_', ' ')}")
        if self.product_asin:
            title = products_map.get(self.product_asin, self.product_asin)
            pills.append(f"Product: {title[:25]}...")
        if self.rating_min > 1.0 or self.rating_max < 5.0:
            pills.append(f"Rating: {self.rating_min:.1f}–{self.rating_max:.1f}★")
        if self.sentiment:
            pills.append(f"Sentiment: {self.sentiment}")
        if self.verified_only:
            pills.append("Verified only")
        if self.date_start and self.date_end:
            pills.append(f"Dates: {self.date_start} to {self.date_end}")
        return pills


class CorpusFilterService:
    def __init__(self, evidence_service: Optional[ProductEvidenceService] = None):
        self.evidence_service = evidence_service or ProductEvidenceService()

    def get_products_for_category(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return products matching the given category (or all if category is None)."""
        return self.evidence_service.get_all_products(category_filter=category)

    def filter_data(self, state: FilterState) -> Dict[str, Any]:
        """
        Execute deterministic filtering across all products and indexed reviews.
        Returns unified metrics, filtered products, filtered reviews, and rating distributions.
        """
        # Step 1: Filter products by category
        candidate_products = self.evidence_service.get_all_products(category_filter=state.category)

        # Step 2: Filter by specific product if selected
        if state.product_asin:
            candidate_products = [
                p for p in candidate_products
                if (p.get("parent_asin") or p.get("asin")) == state.product_asin
            ]

        # Step 3: Collect and filter reviews
        filtered_reviews = []
        product_review_counts = {}

        # Parse date boundaries if specified
        start_ts = None
        end_ts = None
        if state.date_start:
            try:
                start_ts = datetime.datetime.strptime(state.date_start, "%Y-%m-%d").timestamp() * 1000
            except Exception:
                pass
        if state.date_end:
            try:
                # Include entire end date
                end_ts = (datetime.datetime.strptime(state.date_end, "%Y-%m-%d") + datetime.timedelta(days=1)).timestamp() * 1000
            except Exception:
                pass

        for p in candidate_products:
            asin = p.get("parent_asin") or p.get("asin")
            revs = self.evidence_service.get_product_reviews(asin)
            p_matching = []

            for r in revs:
                # Rating filter (inclusive)
                rating = r.get("rating")
                if rating is not None:
                    try:
                        r_val = float(rating)
                        if r_val < state.rating_min or r_val > state.rating_max:
                            continue
                    except Exception:
                        continue
                elif state.rating_min > 1.0 or state.rating_max < 5.0:
                    continue

                # Verified purchase filter
                if state.verified_only:
                    if not r.get("verified_purchase", False):
                        continue

                # Date filter
                r_ts = r.get("timestamp")
                if r_ts is not None:
                    try:
                        ts_num = float(r_ts)
                        if start_ts and ts_num < start_ts:
                            continue
                        if end_ts and ts_num > end_ts:
                            continue
                    except Exception:
                        pass

                # Sentiment filter
                if state.sentiment and state.sentiment != "All Sentiment":
                    r_sent = str(r.get("sentiment", "")).capitalize()
                    # If not explicitly tagged, infer from rating: 4-5=Positive, 3=Neutral, 1-2=Negative
                    if not r_sent or r_sent == "None":
                        r_rating = float(r.get("rating", 3.0))
                        if r_rating >= 4.0:
                            r_sent = "Positive"
                        elif r_rating == 3.0:
                            r_sent = "Neutral"
                        else:
                            r_sent = "Negative"
                    if r_sent != state.sentiment:
                        continue

                p_matching.append(r)

            product_review_counts[asin] = len(p_matching)
            filtered_reviews.extend(p_matching)

        # Products with matching reviews (or candidate products if no review filter applied)
        if state.is_default():
            active_products = candidate_products
        else:
            # Keep products that have matching reviews, or if no reviews remain, keep candidate product
            active_products = [
                p for p in candidate_products
                if product_review_counts.get(p.get("parent_asin") or p.get("asin"), 0) > 0 or len(candidate_products) == 1
            ]

        # Calculate metrics
        total_revs = len(filtered_reviews)
        verified_count = sum(1 for r in filtered_reviews if r.get("verified_purchase", True))
        verified_pct = round((verified_count / total_revs * 100), 1) if total_revs > 0 else 100.0

        # Star breakdown (1★ through 5★)
        star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in filtered_reviews:
            try:
                stars = int(round(float(r.get("rating", 5))))
                if 1 <= stars <= 5:
                    star_counts[stars] += 1
            except Exception:
                pass

        # Per-product rating comparison
        product_ratings = []
        for p in candidate_products:
            asin = p.get("parent_asin") or p.get("asin")
            p_revs = [r for r in filtered_reviews if (r.get("parent_asin") or r.get("asin")) == asin]
            if p_revs:
                avg_r = round(sum(float(r.get("rating", 0)) for r in p_revs) / len(p_revs), 1)
                count = len(p_revs)
            else:
                avg_r = round(float(p.get("average_rating", 0.0)), 1)
                count = int(p.get("rating_number", 0))
            
            product_ratings.append({
                "asin": asin,
                "title": p.get("title", "Product"),
                "short_title": p.get("title", "Product")[:22] + "..." if len(p.get("title", "")) > 25 else p.get("title", ""),
                "average_rating": avg_r,
                "sample_size": count,
                "category": p.get("main_category", "Unknown")
            })

        # Sort product ratings by rating descending
        product_ratings.sort(key=lambda x: (x["average_rating"], x["sample_size"]), reverse=True)

        # Observation window
        timestamps = [
            float(r.get("timestamp")) for r in filtered_reviews
            if r.get("timestamp") and float(r.get("timestamp", 0)) > 0
        ]
        if timestamps:
            min_dt = datetime.datetime.fromtimestamp(min(timestamps) / 1000)
            max_dt = datetime.datetime.fromtimestamp(max(timestamps) / 1000)
            window_str = f"{min_dt.strftime('%Y')} Q{(min_dt.month-1)//3 + 1} – Q{(max_dt.month-1)//3 + 1}"
        else:
            window_str = "2023 Q1 – Q4"

        # Unique categories represented
        categories_count = len(set(p.get("main_category") for p in active_products if p.get("main_category")))

        # Tracked defects in active products
        tracked_defects = 0
        nlp_cache = getattr(self.evidence_service.loader, "_cached_nlp_insights", {})
        for p in active_products:
            asin = p.get("parent_asin") or p.get("asin")
            ins = nlp_cache.get(asin, {})
            tracked_defects += len(ins.get("issues", []))

        return {
            "total_reviews": total_revs,
            "verified_ratio_pct": verified_pct,
            "products_cataloged": len(active_products),
            "categories_represented": categories_count,
            "emerging_issues": tracked_defects,
            "observation_window": window_str,
            "star_counts": star_counts,
            "product_ratings": product_ratings,
            "active_products": active_products,
            "filtered_reviews": filtered_reviews
        }
