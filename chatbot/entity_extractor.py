"""
Entity Extractor for Product Names, Categories, and Aspects.
Uses fuzzy string matching (rapidfuzz), regex boundaries, and domain dictionaries to accurately
identify products, categories, and focus aspects even with typos or colloquial abbreviations.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from rapidfuzz import fuzz, process

from config import CATEGORIES
from utils.logger import get_logger

logger = get_logger("entity_extractor")

# Product aliases to map common short names to canonical parent_asins
PRODUCT_ALIASES = {
    "B09XS7JWHH": ["sony wh-1000xm5", "sony xm5", "xm5", "wh-1000xm5", "sony headphones", "sony", "wh1000xm5", "sony headset", "sonyin"],
    "B0BDHW4P5S": ["airpods pro", "airpods pro 2", "apple airpods", "airpods", "apple earbuds", "airpod", "air pods"],
    "B08N5M7S6K": ["logitech mx master 3s", "mx master 3s", "mx master", "logitech mouse", "logitech", "mx master 3"],
    "B00FLYWNYQ": ["instant pot duo", "instant pot", "electric pressure cooker", "instant pot 6 quart", "pressure cooker"],
    "B07FB15XBG": ["keurig k-mini", "keurig", "k-mini", "keurig coffee maker", "k mini", "coffee maker"],
    "B07S76WBJ1": ["ninja professional blender", "ninja blender", "ninja auto-iq", "ninja"],
    "B01MSSDEPK": ["cerave hydrating cleanser", "cerave facial cleanser", "cerave cleanser", "cerave"],
    "B089KV92K9": ["dyson airwrap", "airwrap", "dyson multi-styler", "dyson hair styler", "dyson"],
    "B000142FVW": ["revlon one-step", "revlon volumizer", "revlon hair dryer brush", "revlon brush", "revlon"]
}

ASPECT_KEYWORDS = {
    "sound": ["sound", "audio", "bass", "treble", "highs", "clarity", "volume", "music"],
    "anc": ["anc", "noise cancellation", "noise canceling", "transparency", "isolation"],
    "comfort": ["comfort", "comfortable", "headband", "earcups", "fit", "ergonomics", "pain", "skull", "ear tips"],
    "battery": ["battery", "battery life", "charging", "charge", "playtime", "drain"],
    "build": ["build", "durability", "plastic", "hinge", "materials", "craftsmanship"],
    "price": ["price", "cost", "value", "expensive", "cheap", "worth it", "dollars"],
    "pump": ["pump", "dispense", "water reservoir", "water flow", "brewing"],
    "thermal": ["heat", "burn", "overheat", "temperature", "scorch"]
}

class EntityExtractor:
    def __init__(self, products: Optional[List[Dict[str, Any]]] = None):
        self.products = products or []
        self._build_product_index()

    def update_products(self, products: List[Dict[str, Any]]):
        """Update indexed product list."""
        self.products = products
        self._build_product_index()

    def _build_product_index(self):
        """Build search corpus for matching."""
        self.index_entries = [] # tuples of (search_phrase, asin, product_dict)

        for p in self.products:
            asin = p.get("parent_asin") or p.get("asin", "")
            title = p.get("title", "")
            self.index_entries.append((title.lower(), asin, p))

            # Add known aliases
            if asin in PRODUCT_ALIASES:
                for alias in PRODUCT_ALIASES[asin]:
                    self.index_entries.append((alias.lower(), asin, p))

    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract matched product(s), category, and key aspect focus from the user query."""
        q_lower = query.lower()

        # 1. Match category
        matched_category = None
        for cat in CATEGORIES:
            cat_clean = cat.replace("_", " ").lower()
            if cat_clean in q_lower:
                matched_category = cat
                break
        if not matched_category:
            if any(w in q_lower for w in ["headphone", "earbud", "mouse", "electronic", "gadget"]):
                matched_category = "Electronics"
            elif any(w in q_lower for w in ["cooker", "coffee", "blender", "kitchen", "pot"]):
                matched_category = "Home_and_Kitchen"
            elif any(w in q_lower for w in ["cleanser", "hair", "skin", "beauty", "styler"]):
                matched_category = "Beauty_and_Personal_Care"

        # 2. Match products (can match multiple for comparisons)
        matched_products = []
        seen_asins = set()

        if self.index_entries:
            # Check for direct phrase matches with word boundaries or exact substring
            for phrase, asin, prod in self.index_entries:
                # Use regex word boundaries for short aliases like 'xm5', 'sony', 'ninja'
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, q_lower) and asin not in seen_asins:
                    matched_products.append(prod)
                    seen_asins.add(asin)

            # If no direct match, check fuzzy matching
            if not matched_products:
                choices = [entry[0] for entry in self.index_entries]
                best_match = process.extractOne(q_lower, choices, scorer=fuzz.partial_ratio)
                if best_match and best_match[1] >= 70:
                    matched_phrase = best_match[0]
                    for entry in self.index_entries:
                        if entry[0] == matched_phrase and entry[1] not in seen_asins:
                            matched_products.append(entry[2])
                            seen_asins.add(entry[1])
                            break

        primary_product = matched_products[0] if matched_products else None

        # 3. Match focus aspect
        matched_aspect = None
        for aspect_name, kws in ASPECT_KEYWORDS.items():
            if any(re.search(r"\b" + re.escape(kw) + r"\b", q_lower) for kw in kws):
                matched_aspect = aspect_name
                break

        # 4. Check comparison intent boundary
        is_comparison = len(matched_products) > 1 or bool(re.search(r"\b(?:vs\.?|versus|compare|comparison|better\s+than)\b", q_lower))

        return {
            "query": query,
            "matched_category": matched_category,
            "primary_product": primary_product,
            "all_matched_products": matched_products,
            "matched_aspect": matched_aspect,
            "is_comparison": is_comparison
        }
