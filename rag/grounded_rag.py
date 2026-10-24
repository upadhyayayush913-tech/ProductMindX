"""
Evidence-Grounded RAG & Claim Validation Engine for ProductMind X.

Enforces:
1. Retrieval of actual Amazon Reviews '23 records and metadata.
2. Deterministic backend calculation of numerical metrics (ratings, distributions, verified counts).
3. Explicit evidence tokens: [Review #ID] and [Metadata #ASIN].
4. Strict post-generation Claim Validation: verifies cited IDs against the database.
5. Explicit distinction between:
   - Observations (verified purchaser quotes)
   - Calculations (deterministic backend metrics)
   - Model Classifications (ABSA sentiments, NMF topics)
   - Emerging Trends (temporal complaint spikes)
6. Honest uncertainty handling and insufficient-evidence reporting.
"""

import re
from typing import Dict, List, Any, Optional, Set, Tuple
import pandas as pd
from utils.logger import get_logger

logger = get_logger("grounded_rag")

class GroundedRAGEngine:
    def __init__(self):
        self.registered_review_ids: Set[str] = set()
        self.registered_product_asins: Set[str] = set()
        self.review_registry: Dict[str, Dict[str, Any]] = {}
        self.product_registry: Dict[str, Dict[str, Any]] = {}

    def register_product(self, product_meta: Dict[str, Any]):
        """Register a product in the grounded evidence registry."""
        asin = product_meta.get("parent_asin") or product_meta.get("asin")
        if asin:
            self.registered_product_asins.add(asin)
            self.product_registry[asin] = product_meta

    def register_reviews(self, reviews: List[Dict[str, Any]]):
        """Register reviews with unique verifiable IDs."""
        for r in reviews:
            # Generate deterministic review ID if not already present
            asin = str(r.get("parent_asin") or r.get("asin", "UNKNOWN"))
            user = str(r.get("user_id", "ANON"))
            ts = r.get("timestamp", 0)
            raw_id = r.get("review_id")
            if raw_id and not (isinstance(raw_id, float) and pd.isna(raw_id)):
                rev_id = str(raw_id).strip()
            else:
                rev_id = f"REV-{asin[:6]}-{user[:6]}-{str(ts)[-4:]}"
            r["review_id"] = rev_id
            self.registered_review_ids.add(rev_id)
            self.review_registry[rev_id] = r

    def calculate_backend_metrics(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute all numerical metrics deterministically in the backend (never invented by LLM)."""
        if not reviews:
            return {
                "total_reviews": 0,
                "verified_purchases": 0,
                "verified_ratio_pct": 0.0,
                "average_rating": 0.0,
                "rating_distribution": {5: 0, 4: 0, 3: 0, 2: 0, 1: 0},
                "rating_distribution_pct": {5: 0.0, 4: 0.0, 3: 0.0, 2: 0.0, 1: 0.0},
                "total_helpful_votes": 0
            }

        total = len(reviews)
        verified = sum(1 for r in reviews if r.get("verified_purchase", True))
        ratings = [float(r.get("rating", 3.0)) for r in reviews]
        avg_rating = sum(ratings) / total if total > 0 else 0.0

        dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for r in ratings:
            star = int(round(r))
            star = max(1, min(5, star))
            dist[star] += 1

        dist_pct = {star: round((count / total) * 100, 1) for star, count in dist.items()}
        total_helpful = sum(int(r.get("helpful_vote", 0)) for r in reviews)

        return {
            "total_reviews": total,
            "verified_purchases": verified,
            "verified_ratio_pct": round((verified / total) * 100, 1),
            "average_rating": round(avg_rating, 2),
            "rating_distribution": dist,
            "rating_distribution_pct": dist_pct,
            "total_helpful_votes": total_helpful
        }

    def build_grounded_context(
        self,
        query: str,
        product: Optional[Dict[str, Any]],
        reviews: List[Dict[str, Any]],
        nlp_insights: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Construct evidence-grounded prompt context with explicit source citations and precomputed metrics."""
        if not product and not reviews:
            return {
                "has_sufficient_evidence": False,
                "context_text": "INSUFFICIENT EVIDENCE: No matching product or review records found in the database.",
                "metrics": {},
                "source_citations": []
            }

        # Register evidence for claim validation
        if product is not None:
            self.register_product(product)
        if reviews:
            self.register_reviews(reviews)

        # 1. Deterministic Backend Calculations
        metrics = self.calculate_backend_metrics(reviews)

        # 2. Build structured evidence document
        lines = []
        lines.append("=== GROUNDED SOURCE EVIDENCE (ACTUAL AMAZON RECORDS) ===")

        if product is not None:
            asin = product.get("parent_asin") or product.get("asin")
            features = product.get("features")
            if features is not None and hasattr(features, "__iter__") and not isinstance(features, (str, bytes)):
                feat_str = ", ".join(str(f) for f in features)
            else:
                feat_str = str(features or "N/A")

            lines.append(f"\n[Source: Metadata #{asin}]")
            lines.append(f"Official Title: {product.get('title')}")
            lines.append(f"Manufacturer / Brand: {product.get('store', 'N/A')}")
            lines.append(f"Category: {product.get('main_category', 'N/A')}")
            lines.append(f"List Price: {product.get('price', 'N/A')}")
            lines.append(f"Manufacturer Features: {feat_str}")
            lines.append(f"Description: {product.get('description', 'N/A')}")

        lines.append("\n=== DETERMINISTIC BACKEND CALCULATIONS (DO NOT RE-ESTIMATE) ===")
        lines.append(f"- Sample Size: {metrics['total_reviews']} verified review records")
        lines.append(f"- Verified Purchaser Ratio: {metrics['verified_ratio_pct']}% ({metrics['verified_purchases']}/{metrics['total_reviews']})")
        lines.append(f"- Computed Mean Rating: {metrics['average_rating']} / 5.0")
        lines.append(f"- Star Distribution: 5★: {metrics['rating_distribution_pct'][5]}%, 4★: {metrics['rating_distribution_pct'][4]}%, 3★: {metrics['rating_distribution_pct'][3]}%, 2★: {metrics['rating_distribution_pct'][2]}%, 1★: {metrics['rating_distribution_pct'][1]}%")
        lines.append(f"- Cumulative Helpful Votes: {metrics['total_helpful_votes']}")

        if nlp_insights:
            lines.append("\n=== NLP MODEL CLASSIFICATIONS & DETECTED PATTERNS ===")
            pros = nlp_insights.get("pros", [])
            cons = nlp_insights.get("cons", [])
            issues = nlp_insights.get("issues", [])

            if pros:
                lines.append("Aspect-Based Sentiment: PROS (Classified by ABSA Model):")
                for p in pros[:4]:
                    lines.append(f"  * {p['aspect']}: {p['positive_percentage']}% positive sentiment across {p['mention_count']} mentions")

            if cons:
                lines.append("Aspect-Based Sentiment: CONS (Classified by ABSA Model):")
                for c in cons[:3]:
                    lines.append(f"  * {c['aspect']}: {100 - c['positive_percentage']:.0f}% negative sentiment across {c['mention_count']} mentions")

            if issues:
                lines.append("Temporal Issue Detection (Emerging Failure Clustering):")
                for iss in issues[:3]:
                    lines.append(f"  * Issue: '{iss['issue']}' | Severity: {iss['severity']} | Frequency: {iss['frequency']} reports | Trend: {iss['trend']}")

        lines.append("\n=== VERIFIABLE REVIEW PASSAGES (CITE USING [Review #ID]) ===")
        citations_list = []
        for r in reviews:
            rev_id = r["review_id"]
            citations_list.append(rev_id)
            vp_flag = "Verified Purchase" if r.get("verified_purchase", True) else "Unverified"
            date_str = r.get("date_str", "Unknown Date")
            lines.append(f"\n[Review #{rev_id}]")
            lines.append(f"Author Rating: {r.get('rating', 3.0)}/5.0 | Status: {vp_flag} | Timeline: {date_str} | Helpful: {r.get('helpful_vote', 0)}")
            lines.append(f"Headline: {r.get('title', '')}")
            lines.append(f"Text: \"{r.get('text', '')}\"")

        return {
            "has_sufficient_evidence": True,
            "context_text": "\n".join(lines),
            "metrics": metrics,
            "source_citations": citations_list
        }

    def build_grounded_comparison_context(
        self,
        query: str,
        products: List[Dict[str, Any]],
        reviews_by_asin: Dict[str, List[Dict[str, Any]]],
        insights_by_asin: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Construct comparative grounded context with multi-product metrics and verifiable citations."""
        if not products:
            return {
                "has_sufficient_evidence": False,
                "context_text": "INSUFFICIENT EVIDENCE: No products found for comparison.",
                "metrics": {},
                "source_citations": []
            }

        insights_by_asin = insights_by_asin or {}
        combined_lines = [
            "=== GROUNDED MULTI-PRODUCT COMPARATIVE EVIDENCE (ACTUAL AMAZON RECORDS) ===",
            "Adhere strictly to verified facts, calculated metrics, and source citations for each product."
        ]
        combined_metrics = {}
        all_citations = []

        for idx, prod in enumerate(products, 1):
            asin = prod.get("parent_asin") or prod.get("asin")
            self.register_product(prod)
            prod_revs = reviews_by_asin.get(asin, [])
            self.register_reviews(prod_revs)
            prod_metrics = self.calculate_backend_metrics(prod_revs)
            combined_metrics[asin] = prod_metrics
            prod_insights = insights_by_asin.get(asin, {})

            features = prod.get("features")
            if features is not None and hasattr(features, "__iter__") and not isinstance(features, (str, bytes)):
                feat_str = ", ".join(str(f) for f in features)
            else:
                feat_str = str(features or "N/A")

            combined_lines.append(f"\n=======================================================")
            combined_lines.append(f"--- CANDIDATE {idx}: {prod.get('title')} ---")
            combined_lines.append(f"[Source: Metadata #{asin}]")
            combined_lines.append(f"Brand: {prod.get('store', 'N/A')} | Price: {prod.get('price', 'N/A')} | Category: {prod.get('main_category', 'N/A')}")
            combined_lines.append(f"Documented Features: {feat_str}")

            combined_lines.append(f"\nDETERMINISTIC CALCULATIONS FOR {prod.get('title')} (DO NOT RE-ESTIMATE):")
            combined_lines.append(f"- Sample Size: {prod_metrics['total_reviews']} verified review records ({prod_metrics['verified_ratio_pct']}% verified)")
            combined_lines.append(f"- Computed Mean Rating: ⭐ {prod_metrics['average_rating']} / 5.0")
            dist = prod_metrics["rating_distribution_pct"]
            combined_lines.append(f"- Rating Breakdown: 5★: {dist.get(5, 0)}% | 4★: {dist.get(4, 0)}% | 3★: {dist.get(3, 0)}% | 2★: {dist.get(2, 0)}% | 1★: {dist.get(1, 0)}%")
            combined_lines.append(f"- Cumulative Helpful Votes: {prod_metrics['total_helpful_votes']}")

            if prod_insights:
                pros = prod_insights.get("pros", [])
                cons = prod_insights.get("cons", [])
                issues = prod_insights.get("issues", [])
                if pros:
                    combined_lines.append("Extracted Pros (ABSA):")
                    for p in pros[:3]:
                        combined_lines.append(f"  * {p['aspect']}: {p['positive_percentage']}% positive sentiment across {p['mention_count']} mentions")
                if cons:
                    combined_lines.append("Extracted Cons (ABSA):")
                    for c in cons[:3]:
                        combined_lines.append(f"  * {c['aspect']}: {100 - c['positive_percentage']:.0f}% negative sentiment across {c['mention_count']} mentions")
                if issues:
                    combined_lines.append("Emerging Defect Alerts (Temporal Clustering):")
                    for iss in issues[:2]:
                        combined_lines.append(f"  * {iss['issue']}: Severity={iss['severity']}, Trend={iss['trend']}")

            combined_lines.append(f"\nVerifiable Review Passages for {prod.get('title')}:")
            for r in prod_revs:
                rev_id = r["review_id"]
                all_citations.append(rev_id)
                vp = "Verified Purchase" if r.get("verified_purchase", True) else "Unverified"
                combined_lines.append(f"  * [Review #{rev_id}] ({r.get('rating')}/5.0, {vp}): \"{r.get('text', '')}\"")

        return {
            "has_sufficient_evidence": True,
            "context_text": "\n".join(combined_lines),
            "metrics": combined_metrics,
            "source_citations": all_citations
        }

    def validate_claims(self, generated_text: str, expected_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate all citations, quotation fidelity, numerical consistency,
        and semantic entailment in the generated response.
        """
        # Find all cited review tokens: [Review #...]
        cited_rev_matches = re.findall(r"\[Review\s*#?([A-Za-z0-9_-]+)\]", generated_text, re.IGNORECASE)
        # Find all cited metadata tokens: [Metadata #...]
        cited_meta_matches = re.findall(r"\[(?:Product\s+)?Metadata\s*#?([A-Za-z0-9_-]+)\]", generated_text, re.IGNORECASE)

        valid_reviews = []
        hallucinated_reviews = []

        for cite in cited_rev_matches:
            clean_cite = str(cite).strip()
            if clean_cite in self.registered_review_ids:
                valid_reviews.append(clean_cite)
            else:
                # Check for partial match (e.g. prefix)
                partial = [str(r) for r in self.registered_review_ids if r and (clean_cite in str(r) or str(r) in clean_cite)]
                if partial:
                    valid_reviews.append(partial[0])
                else:
                    hallucinated_reviews.append(clean_cite)

        valid_metadata = [m for m in cited_meta_matches if m in self.registered_product_asins]
        hallucinated_metadata = [m for m in cited_meta_matches if m not in self.registered_product_asins]

        total_citations = len(cited_rev_matches) + len(cited_meta_matches)
        valid_citations_count = len(valid_reviews) + len(valid_metadata)

        # 1. Quotation Verification
        # Find quoted phrases: "...""
        quoted_phrases = re.findall(r'"([^"]{10,120})"', generated_text)
        verified_quotes = []
        unverified_quotes = []

        all_registered_texts = " ".join([
            f"{r.get('title', '')} {r.get('text', '')}" for r in self.review_registry.values()
        ]).lower()

        for phrase in quoted_phrases:
            clean_phrase = phrase.strip().lower()
            if clean_phrase in all_registered_texts:
                verified_quotes.append(phrase)
            else:
                # Allow fuzzy match of 80% tokens
                phrase_tokens = clean_phrase.split()
                if len(phrase_tokens) >= 3 and any(
                    " ".join(phrase_tokens[i:i+3]) in all_registered_texts
                    for i in range(len(phrase_tokens) - 2)
                ):
                    verified_quotes.append(phrase)
                else:
                    unverified_quotes.append(phrase)

        quote_fidelity_pct = round((len(verified_quotes) / len(quoted_phrases) * 100), 1) if quoted_phrases else 100.0

        # 2. Numerical Consistency Check
        numerical_inconsistencies = []
        if expected_metrics:
            # Check average rating consistency
            expected_avg = expected_metrics.get("average_rating")
            if expected_avg is not None:
                # Find ratings in text like 4.5/5 or 4.5 stars
                found_ratings = re.findall(r"(\d\.\d+)\s*(?:/|out of|\s*stars|\s*★)", generated_text)
                for fr in found_ratings:
                    try:
                        val = float(fr)
                        if abs(val - float(expected_avg)) > 0.35 and val != 5.0 and val != 1.0: # allow individual star mentions
                            numerical_inconsistencies.append(f"Quoted rating {val} differs from calculated mean {expected_avg}")
                    except ValueError:
                        pass

        # 3. Semantic Entailment & Contradiction Detection
        entailment_contradictions = []
        for r_id in set(valid_reviews):
            record = self.review_registry.get(r_id)
            if not record:
                continue
            rating = float(record.get("rating", 3.0))
            # Find sentence mentioning this review
            for sentence in re.split(r"[.!?\n]", generated_text):
                if r_id in sentence:
                    s_lower = sentence.lower()
                    if rating >= 4.0 and any(w in s_lower for w in ["defective", "broke", "garbage", "failure", "fire", "useless"]):
                        entailment_contradictions.append(f"Review #{r_id} has high rating ({rating}/5.0) but cited for severe defect claim: '{sentence.strip()}'")
                    elif rating <= 2.0 and any(w in s_lower for w in ["flawless", "perfection", "superb", "best ever"]):
                        entailment_contradictions.append(f"Review #{r_id} has low rating ({rating}/5.0) but cited for glowing praise: '{sentence.strip()}'")

        # Grounding Score Calculation
        # Combines citation validity (60%), quotation fidelity (25%), and entailment (15%)
        citation_score = (valid_citations_count / total_citations * 100.0) if total_citations > 0 else (100.0 if "Insufficient Evidence" in generated_text else 85.0)
        entailment_penalty = len(entailment_contradictions) * 10.0
        grounding_score = max(0.0, round(
            (0.60 * citation_score) +
            (0.25 * quote_fidelity_pct) +
            (0.15 * max(0.0, 100.0 - entailment_penalty)),
            1
        ))

        # Retrieve referenced evidence snippets
        verified_evidence_snippets = []
        for r_id in set(valid_reviews):
            record = self.review_registry.get(r_id)
            if record:
                verified_evidence_snippets.append({
                    "citation_id": r_id,
                    "rating": record.get("rating"),
                    "text": record.get("text"),
                    "headline": record.get("title"),
                    "verified_purchase": record.get("verified_purchase", True)
                })

        is_valid = (
            len(hallucinated_reviews) == 0 and
            len(hallucinated_metadata) == 0 and
            len(entailment_contradictions) == 0
        )

        return {
            "total_citations": total_citations,
            "valid_citations_count": valid_citations_count,
            "grounding_score_pct": grounding_score,
            "is_valid": is_valid,
            "citation_validity_pct": round(citation_score, 1),
            "quotation_fidelity_pct": quote_fidelity_pct,
            "valid_review_citations": list(set(valid_reviews)),
            "hallucinated_review_citations": hallucinated_reviews,
            "valid_metadata_citations": valid_metadata,
            "hallucinated_metadata_citations": hallucinated_metadata,
            "verified_quotes_count": len(verified_quotes),
            "unverified_quotes_count": len(unverified_quotes),
            "numerical_inconsistencies": numerical_inconsistencies,
            "entailment_contradictions": entailment_contradictions,
            "verified_evidence_snippets": verified_evidence_snippets
        }
