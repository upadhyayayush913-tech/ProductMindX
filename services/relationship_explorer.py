"""
Product Issue Relationship Explorer Service for ProductMind X.

Constructs an in-memory bipartite/tripartite relationship graph connecting:
Product <---> Aspect <---> Issue Phrase <---> Evidence Record

Enables relational navigation without external graph databases:
- From Issue -> Affected Products & Aspect
- From Product -> Aspects & Issue Clusters
- From Finding -> Exact Review Excerpts and Citation IDs.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict
import re

from services.evidence_contract import EvidenceRecord
from nlp_pipeline.sentiment_lexicon import ASPECT_TAXONOMY, STRONG_NEGATIVE_WORDS
from utils.logger import get_logger

logger = get_logger("relationship_explorer")

# Common product complaint phrases by aspect
KNOWN_ISSUE_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "battery life": {
        "Rapid Battery Drain": ["drain", "draining", "dies", "dead", "short battery", "burn through"],
        "Charging Failure": ["charging", "won't charge", "refuses to charge", "loose port", "stopped charging"],
        "Battery Degradation": ["degraded", "capacity", "holding charge", "overheats while charging"]
    },
    "sound quality": {
        "Audio Distortion & Static": ["distort", "static", "buzzing", "crackling", "hissing", "scratchy"],
        "Muffled Microphones": ["muffled", "tinny", "low volume", "hollow", "poor mic"],
        "Treble Harshness": ["screeching", "harsh", "piercing", "painful highs"]
    },
    "noise cancellation": {
        "Weak Ambient Isolation": ["barely blocks", "humming", "weak anc", "cabin pressure"],
        "High-Pitched Whine": ["whine", "screech", "pressure headache", "popping noise"]
    },
    "comfort & fit": {
        "Clamping Force Pressure": ["clamping", "headache", "pinching", "tight", "ear pain"],
        "Ear Fatigue & Sweating": ["shallow earcups", "fatigue", "hot", "sweaty", "cartilage"]
    },
    "build quality": {
        "Hinge & Plastic Snapping": ["hinge snapped", "cracked", "flimsy", "plastic broke", "rattles"],
        "Slider Durability Failure": ["headband snapped", "broken slider", "creaking", "durability"]
    },
    "software & connectivity": {
        "Bluetooth Disconnections": ["disconnects", "drops calls", "pairing failure", "connection drops"],
        "App Sync Crash": ["app crashes", "firmware bug", "fails to detect", "bluetooth latency"]
    }
}


class ProductIssueRelationshipExplorer:
    def __init__(self, loader=None):
        from data.data_loader import AmazonDataLoader
        self.loader = loader or AmazonDataLoader()
        self._graph_built = False
        self.issues: Dict[str, Dict[str, Any]] = {}
        self.product_issues: Dict[str, List[str]] = defaultdict(list)
        self.aspect_issues: Dict[str, List[str]] = defaultdict(list)
        self.issue_evidence: Dict[str, List[EvidenceRecord]] = defaultdict(list)

    def build_graph(self):
        """Construct the in-memory relationship graph from products and reviews."""
        if self._graph_built:
            return

        all_prods = self.loader.get_all_products()
        for prod in all_prods:
            asin = prod.get("parent_asin") or prod.get("asin")
            title = prod.get("title", "Product")
            cat = prod.get("main_category", "Electronics")

            # Load reviews for this product
            try:
                rev_df = self.loader.load_reviews(cat)
                p_revs = rev_df[rev_df["parent_asin"] == asin].to_dict("records") if not rev_df.empty else []
            except Exception as e:
                logger.warning(f"Error loading reviews for {asin}: {e}")
                p_revs = []

            for r in p_revs:
                text = str(r.get("text", "")).lower()
                rev_id = str(r.get("review_id", f"REV-{asin[:6]}-{str(r.get('timestamp', 0))[-4:]}"))
                rating = float(r.get("rating", 3.0))

                for aspect, issue_map in KNOWN_ISSUE_TAXONOMY.items():
                    for issue_name, keywords in issue_map.items():
                        if any(kw in text for kw in keywords):
                            if issue_name not in self.issues:
                                self.issues[issue_name] = {
                                    "issue_name": issue_name,
                                    "aspect": aspect,
                                    "affected_products": set(),
                                    "total_complaint_mentions": 0,
                                    "severity": "HIGH" if rating <= 2.0 else "MEDIUM"
                                }

                            self.issues[issue_name]["affected_products"].add(title)
                            self.issues[issue_name]["total_complaint_mentions"] += 1
                            if issue_name not in self.product_issues[asin]:
                                self.product_issues[asin].append(issue_name)
                            if issue_name not in self.aspect_issues[aspect]:
                                self.aspect_issues[aspect].append(issue_name)

                            # Create evidence record
                            ev = EvidenceRecord(
                                review_id=rev_id,
                                product_asin=asin,
                                product_title=title,
                                category=cat,
                                aspect=aspect,
                                excerpt=str(r.get("text", ""))[:260] + "...",
                                rating=rating,
                                timestamp=int(r.get("timestamp", 0)) if r.get("timestamp") else 0,
                                verified_purchase=bool(r.get("verified_purchase", True)),
                                retrieval_method="relational_graph",
                                retrieval_score=1.0,
                                supports_claim=rating <= 2.0,
                                validation_state="VERIFIED"
                            )
                            self.issue_evidence[issue_name].append(ev)

        self._graph_built = True
        logger.info(f"Built relationship graph: {len(self.issues)} issue clusters mapped.")

    def get_all_issues(self) -> List[Dict[str, Any]]:
        """Return list of all mapped issue clusters."""
        self.build_graph()
        result = []
        for name, data in self.issues.items():
            result.append({
                "issue_name": name,
                "aspect": data["aspect"],
                "affected_products_count": len(data["affected_products"]),
                "affected_products": sorted(list(data["affected_products"])),
                "total_complaint_mentions": data["total_complaint_mentions"],
                "severity": data["severity"]
            })
        return sorted(result, key=lambda x: x["total_complaint_mentions"], reverse=True)

    def get_issue_details(self, issue_name: str) -> Optional[Dict[str, Any]]:
        """Get complete relational profile for an issue."""
        self.build_graph()
        if issue_name not in self.issues:
            return None

        issue_data = self.issues[issue_name]
        evidence = self.issue_evidence.get(issue_name, [])

        return {
            "issue_name": issue_name,
            "aspect": issue_data["aspect"],
            "affected_products": sorted(list(issue_data["affected_products"])),
            "complaint_count": issue_data["total_complaint_mentions"],
            "severity": issue_data["severity"],
            "evidence_records": [ev.to_dict() for ev in evidence]
        }

    def get_product_issues(self, asin: str) -> List[str]:
        """Return all issue clusters affecting a product."""
        self.build_graph()
        return self.product_issues.get(asin, [])


# Global singleton
relationship_explorer = ProductIssueRelationshipExplorer()
