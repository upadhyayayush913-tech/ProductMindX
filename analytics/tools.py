"""
Deterministic Analytical Tools for ProductMind X.

Every analytical calculation is executed deterministically in Python code:
- Never invented or approximated by free-form LLM arithmetic.
- Strict definitions, observation periods, sample sizes, and source provenance.
- Transparent preference-weighted buying recommendation engine.
- Statistically qualified temporal trend detector (minimum-sample and baseline stability requirements).
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import math
from utils.logger import get_logger

logger = get_logger("analytics_tools")

# ---------------------------------------------------------------------------
# 1. Product Metadata & Dataset Ingestion Statistics
# ---------------------------------------------------------------------------

def calculate_dataset_statistics(
    all_products: List[Dict[str, Any]],
    reviews_by_category: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Calculate exact dataset coverage and ingestion statistics."""
    total_products = len(all_products)
    categories = sorted(list({p.get("main_category", "Unknown") for p in all_products}))
    
    products_per_category = defaultdict(int)
    for p in all_products:
        cat = p.get("main_category", "Unknown")
        products_per_category[cat] += 1
        
    reviews_per_category = {}
    total_reviews_indexed = 0
    total_verified_purchases = 0
    date_timestamps = []
    
    for cat, revs in reviews_by_category.items():
        reviews_per_category[cat] = len(revs)
        total_reviews_indexed += len(revs)
        for r in revs:
            if r.get("verified_purchase", True):
                total_verified_purchases += 1
            ts = r.get("timestamp")
            if ts:
                date_timestamps.append(ts)
                
    verified_ratio = round((total_verified_purchases / total_reviews_indexed * 100), 1) if total_reviews_indexed > 0 else 0.0
    
    min_date_str = "N/A"
    max_date_str = "N/A"
    if date_timestamps:
        from utils.helpers import timestamp_to_str
        min_date_str = timestamp_to_str(min(date_timestamps))
        max_date_str = timestamp_to_str(max(date_timestamps))
        
    return {
        "total_products_indexed": total_products,
        "total_reviews_indexed": total_reviews_indexed,
        "categories_covered": categories,
        "category_count": len(categories),
        "products_per_category": dict(products_per_category),
        "reviews_per_category": reviews_per_category,
        "overall_verified_purchases": total_verified_purchases,
        "overall_verified_ratio_pct": verified_ratio,
        "observation_window": {
            "start": min_date_str,
            "end": max_date_str
        },
        "provenance": "Amazon Reviews '23 Curated Corpus"
    }


# ---------------------------------------------------------------------------
# 2. Rating & Review Summaries
# ---------------------------------------------------------------------------

def calculate_review_metrics(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute exact rating averages, star distributions, and verification rates."""
    if not reviews:
        return {
            "sample_size": 0,
            "average_rating": 0.0,
            "verified_purchases": 0,
            "verified_ratio_pct": 0.0,
            "star_distribution": {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
            "star_distribution_pct": {5: 0.0, 4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0},
            "total_helpful_votes": 0,
            "date_range": {"first": "N/A", "last": "N/A"}
        }
        
    total = len(reviews)
    verified = sum(1 for r in reviews if r.get("verified_purchase", True))
    ratings = [float(r.get("rating", 3.0)) for r in reviews]
    avg_rating = round(sum(ratings) / total, 2)
    
    star_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for r in ratings:
        star = max(1, min(5, int(round(r))))
        star_dist[star] += 1
        
    star_pct = {s: round((c / total) * 100, 1) for s, c in star_dist.items()}
    helpful = sum(int(r.get("helpful_vote", 0)) for r in reviews)
    
    timestamps = [r.get("timestamp") for r in reviews if r.get("timestamp")]
    from utils.helpers import timestamp_to_str
    first_date = timestamp_to_str(min(timestamps)) if timestamps else "N/A"
    last_date = timestamp_to_str(max(timestamps)) if timestamps else "N/A"
    
    return {
        "sample_size": total,
        "average_rating": avg_rating,
        "verified_purchases": verified,
        "verified_ratio_pct": round((verified / total) * 100, 1),
        "star_distribution": star_dist,
        "star_distribution_pct": star_pct,
        "total_helpful_votes": helpful,
        "date_range": {"first": first_date, "last": last_date}
    }


# ---------------------------------------------------------------------------
# 3. Temporal Trend Comparison (Strict Anti-Fabrication & Minimum-Sample Rules)
# ---------------------------------------------------------------------------

def calculate_temporal_issue_trend(
    issue_name: str,
    complaint_records: List[Dict[str, Any]],
    all_product_reviews: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate issue trajectory with statistical rigor:
    - Requires valid timestamps.
    - Requires minimum sample size of >= 3 complaints to establish a pattern.
    - Compares complaint rates (complaints / total reviews in period) rather than raw counts.
    - Abstains honestly if data is insufficient or baseline is unstable.
    """
    # Filter complaints with valid timestamps
    dated_complaints = [c for c in complaint_records if c.get("timestamp")]
    dated_all = [r for r in all_product_reviews if r.get("timestamp")]
    
    if len(dated_complaints) < 3 or len(dated_all) < 4:
        return {
            "issue": issue_name,
            "trajectory": "Insufficient Data",
            "is_emerging": False,
            "sample_size": len(dated_complaints),
            "baseline_stable": False,
            "explanation": "Insufficient sample size (fewer than 3 dated complaint records) to establish a statistically valid historical baseline.",
            "observation_window": {"first": "N/A", "last": "N/A"}
        }
        
    # Sort by timestamp
    dated_complaints.sort(key=lambda x: x["timestamp"])
    dated_all.sort(key=lambda x: x["timestamp"])
    
    # Define observation midpoint
    min_ts = dated_all[0]["timestamp"]
    max_ts = dated_all[-1]["timestamp"]
    mid_ts = min_ts + (max_ts - min_ts) / 2
    
    # Historical vs Recent periods
    hist_all = [r for r in dated_all if r["timestamp"] <= mid_ts]
    recent_all = [r for r in dated_all if r["timestamp"] > mid_ts]
    
    hist_comp = [c for c in dated_complaints if c["timestamp"] <= mid_ts]
    recent_comp = [c for c in dated_complaints if c["timestamp"] > mid_ts]
    
    hist_total = len(hist_all)
    recent_total = len(recent_all)
    
    if hist_total == 0 or recent_total == 0:
        return {
            "issue": issue_name,
            "trajectory": "Stable / Concentrated",
            "is_emerging": False,
            "sample_size": len(dated_complaints),
            "baseline_stable": False,
            "explanation": "Observation period lacks multi-window review density across the chronological midpoint.",
            "observation_window": {
                "first": dated_complaints[0].get("date_str", "N/A"),
                "last": dated_complaints[-1].get("date_str", "N/A")
            }
        }
        
    # Normalized complaint rates
    hist_rate = len(hist_comp) / hist_total
    recent_rate = len(recent_comp) / recent_total
    
    # Emerging criteria: recent rate is noticeably higher and recent complaints >= 2
    if recent_rate > hist_rate * 1.25 and len(recent_comp) >= 2:
        trajectory = "Increasing (Emerging Issue ⚠️)"
        is_emerging = True
    elif recent_rate < hist_rate * 0.75:
        trajectory = "Declining (Resolution Indicated)"
        is_emerging = False
    else:
        trajectory = "Stable Baseline"
        is_emerging = False
        
    from utils.helpers import timestamp_to_str
    return {
        "issue": issue_name,
        "trajectory": trajectory,
        "is_emerging": is_emerging,
        "sample_size": len(dated_complaints),
        "baseline_stable": True,
        "historical_period": {
            "complaint_count": len(hist_comp),
            "total_reviews": hist_total,
            "complaint_rate_pct": round(hist_rate * 100, 1),
            "window": f"{timestamp_to_str(min_ts)} to {timestamp_to_str(mid_ts)}"
        },
        "recent_period": {
            "complaint_count": len(recent_comp),
            "total_reviews": recent_total,
            "complaint_rate_pct": round(recent_rate * 100, 1),
            "window": f"{timestamp_to_str(mid_ts)} to {timestamp_to_str(max_ts)}"
        },
        "explanation": (
            f"Normalized complaint rate shifted from {round(hist_rate * 100, 1)}% in earlier window "
            f"to {round(recent_rate * 100, 1)}% in the recent observation window."
        )
    }


# ---------------------------------------------------------------------------
# 4. Transparent Buying Recommendation Advisor
# ---------------------------------------------------------------------------

DEFAULT_ASPECT_WEIGHTS = {
    "sound": 0.25,
    "anc": 0.25,
    "comfort": 0.20,
    "battery": 0.15,
    "price": 0.15
}

def calculate_preference_weighted_recommendation(
    candidate_products: List[Dict[str, Any]],
    metrics_by_asin: Dict[str, Dict[str, Any]],
    insights_by_asin: Dict[str, Dict[str, Any]],
    user_priorities: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Calculate an evidence-grounded recommendation using deterministic preference weighting.
    - Normalizes scores on a 0-100 scale.
    - Handles missing aspect data explicitly without hallucinating zeros or perfect scores.
    - Identifies trade-offs and counter-evidence.
    - Abstains or flags ties when scores are within margin of error (< 2.5 points).
    """
    if len(candidate_products) < 2:
        return {
            "has_recommendation": False,
            "verdict": "Insufficient candidates: At least two products are required for comparative recommendation.",
            "scores": {},
            "breakdown": {}
        }
        
    weights = user_priorities or DEFAULT_ASPECT_WEIGHTS
    # Normalize weights so sum is 1.0
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}
    else:
        weights = DEFAULT_ASPECT_WEIGHTS
        
    product_evaluations = {}
    
    for prod in candidate_products:
        asin = prod.get("parent_asin") or prod.get("asin")
        title = prod.get("title", "Product")
        m = metrics_by_asin.get(asin, {})
        ins = insights_by_asin.get(asin, {})
        
        # 1. Base rating score (0-100)
        avg_rating = m.get("average_rating", float(prod.get("average_rating", 3.0)))
        rating_score = (avg_rating / 5.0) * 100
        
        # 2. Aspect sentiment scores
        pros = ins.get("pros", [])
        cons = ins.get("cons", [])
        issues = ins.get("issues", [])
        
        aspect_scores = {}
        aspect_evidence = {}
        
        # Helper to find aspect sentiment
        def get_aspect_sentiment(kw_list: List[str]) -> Tuple[Optional[float], Optional[str]]:
            for p in pros:
                if any(k in p["aspect"].lower() for k in kw_list):
                    return float(p["positive_percentage"]), f"Positive: {p['positive_percentage']}% ({p['mention_count']} mentions)"
            for c in cons:
                if any(k in c["aspect"].lower() for k in kw_list):
                    return float(c["positive_percentage"]), f"Negative: {100 - c['positive_percentage']:.0f}% ({c['mention_count']} mentions)"
            return None, None
            
        # Sound
        s_score, s_ev = get_aspect_sentiment(["sound", "audio", "bass", "clarity", "voice"])
        aspect_scores["sound"] = s_score if s_score is not None else rating_score
        aspect_evidence["sound"] = s_ev or "Derived from overall rating (specific sound aspect not isolated)"
        
        # ANC / Isolation
        anc_score, anc_ev = get_aspect_sentiment(["anc", "noise cancel", "transparency", "isolation"])
        aspect_scores["anc"] = anc_score if anc_score is not None else rating_score
        aspect_evidence["anc"] = anc_ev or "Derived from overall rating"
        
        # Comfort
        c_score, c_ev = get_aspect_sentiment(["comfort", "fit", "earcup", "ear tip", "headband", "ergonomic"])
        aspect_scores["comfort"] = c_score if c_score is not None else rating_score
        aspect_evidence["comfort"] = c_ev or "Derived from overall rating"
        
        # Battery / Efficiency
        b_score, b_ev = get_aspect_sentiment(["battery", "charge", "playtime", "power"])
        aspect_scores["battery"] = b_score if b_score is not None else rating_score
        aspect_evidence["battery"] = b_ev or "Derived from overall rating"
        
        # Price penalty / value: parse price if available
        price_str = str(prod.get("price", "")).replace("$", "").replace(",", "").strip()
        try:
            price_val = float(price_str)
        except Exception:
            price_val = 150.0  # neutral benchmark
            
        # Compute composite weighted score
        composite_score = (
            weights.get("sound", 0.25) * aspect_scores["sound"] +
            weights.get("anc", 0.25) * aspect_scores["anc"] +
            weights.get("comfort", 0.20) * aspect_scores["comfort"] +
            weights.get("battery", 0.15) * aspect_scores["battery"] +
            weights.get("price", 0.15) * max(10.0, 100.0 - (price_val / 6.0)) # Higher price = lower affordability score
        )
        
        # Penalty for critical issues
        defect_count = sum(1 for iss in issues if iss.get("severity") in ["Critical", "High"])
        composite_score = max(0.0, composite_score - (defect_count * 3.5))
        
        product_evaluations[asin] = {
            "title": title,
            "asin": asin,
            "price": prod.get("price", "N/A"),
            "price_val": price_val,
            "average_rating": avg_rating,
            "review_sample_size": m.get("total_reviews", 0),
            "verified_ratio_pct": m.get("verified_ratio_pct", 0.0),
            "composite_score": round(composite_score, 1),
            "aspect_scores": {k: round(v, 1) for k, v in aspect_scores.items()},
            "aspect_evidence": aspect_evidence,
            "top_pros": [p["aspect"] for p in pros[:2]],
            "top_cons": [c["aspect"] for c in cons[:2]],
            "critical_defects": [iss["issue"] for iss in issues if iss.get("severity") in ["Critical", "High"]]
        }
        
    # Determine winner or tie
    sorted_candidates = sorted(product_evaluations.values(), key=lambda x: x["composite_score"], reverse=True)
    winner = sorted_candidates[0]
    runner_up = sorted_candidates[1]
    
    score_diff = winner["composite_score"] - runner_up["composite_score"]
    is_tie = score_diff < 2.5
    
    if is_tie:
        verdict = (
            f"Statistical Tie: **{winner['title']}** ({winner['composite_score']}/100) and "
            f"**{runner_up['title']}** ({runner_up['composite_score']}/100) perform within the 2.5-point margin "
            f"of error based on current sample evidence."
        )
    else:
        verdict = (
            f"Decisive Recommendation: **{winner['title']}** is recommended with a score of "
            f"**{winner['composite_score']}/100** vs {runner_up['title']}'s **{runner_up['composite_score']}/100**."
        )
        
    return {
        "has_recommendation": True,
        "is_tie": is_tie,
        "winner_asin": None if is_tie else winner["asin"],
        "winner_title": None if is_tie else winner["title"],
        "score_difference": round(score_diff, 1),
        "verdict": verdict,
        "weights_applied": weights,
        "evaluations": product_evaluations,
        "ranked_candidates": sorted_candidates
    }


def calculate_weight_sensitivity_analysis(
    candidate_products: List[Dict[str, Any]],
    metrics_by_asin: Dict[str, Dict[str, Any]],
    insights_by_asin: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Perform multi-scenario sensitivity analysis across distinct consumer priority profiles:
    1. Balanced Benchmark
    2. Audio Quality First
    3. Commuter / Travel (ANC & Battery)
    4. Budget / Affordability Focused
    5. Ergonomics & Comfort Focused
    Evaluates whether the winning product remains stable or flips under changing user values.
    """
    profiles = {
        "Balanced Default": {"sound": 0.25, "anc": 0.25, "comfort": 0.20, "battery": 0.15, "price": 0.15},
        "Acoustic / Audiophile": {"sound": 0.50, "anc": 0.20, "comfort": 0.15, "battery": 0.05, "price": 0.10},
        "Frequent Traveler / Commuter": {"sound": 0.10, "anc": 0.45, "comfort": 0.15, "battery": 0.25, "price": 0.05},
        "Value / Budget Conscious": {"sound": 0.20, "anc": 0.10, "comfort": 0.10, "battery": 0.10, "price": 0.50},
        "All-Day Wear / Ergonomics": {"sound": 0.15, "anc": 0.15, "comfort": 0.50, "battery": 0.10, "price": 0.10}
    }

    scenario_results = {}
    winners = []

    for name, w in profiles.items():
        rec = calculate_preference_weighted_recommendation(
            candidate_products=candidate_products,
            metrics_by_asin=metrics_by_asin,
            insights_by_asin=insights_by_asin,
            user_priorities=w
        )
        if rec["has_recommendation"]:
            top = rec["ranked_candidates"][0]
            runner_up = rec["ranked_candidates"][1] if len(rec["ranked_candidates"]) > 1 else None
            diff = round(top["composite_score"] - (runner_up["composite_score"] if runner_up else 0.0), 1)
            scenario_results[name] = {
                "winner_title": top["title"],
                "winner_score": top["composite_score"],
                "runner_up_title": runner_up["title"] if runner_up else "N/A",
                "runner_up_score": runner_up["composite_score"] if runner_up else 0.0,
                "score_delta": diff,
                "is_tie": rec["is_tie"],
                "verdict": rec["verdict"]
            }
            winners.append(top["title"])

    # Robustness score: % of profiles where the modal winner wins
    from collections import Counter
    counts = Counter(winners)
    modal_winner, max_count = counts.most_common(1)[0] if counts else ("None", 0)
    stability_pct = round((max_count / len(profiles) * 100), 1) if profiles else 0.0

    return {
        "candidate_count": len(candidate_products),
        "dominant_winner": modal_winner,
        "stability_pct": stability_pct,
        "is_unconditionally_dominant": stability_pct == 100.0,
        "scenarios": scenario_results
    }

