"""
Aspect-Based Sentiment Analysis (ABSA) Engine.
Extracts product aspects and opinion terms from review text,
computes aspect-level sentiment scores, and aggregates them into
explainable PROS and CONS for each product.
"""

import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from data.preprocessor import TextPreprocessor
from utils.logger import get_logger

logger = get_logger("aspect_extractor")

# Consumer review domain sentiment terms to enrich standard VADER
CONSUMER_REVIEW_LEXICON = {
    "phenomenal": 3.2,
    "crisp": 2.0,
    "flawless": 3.0,
    "superb": 3.0,
    "unbeatable": 2.8,
    "sturdy": 2.2,
    "durable": 2.5,
    "comfortable": 2.5,
    "lifesaver": 2.8,
    "convenient": 2.0,
    "tactile": 1.8,
    "sleek": 1.8,
    "ergonomic": 2.2,
    "pulverize": 1.5,
    "expensive": -2.0,
    "overpriced": -2.5,
    "fragile": -2.2,
    "crackling": -2.5,
    "buzzing": -2.2,
    "jammed": -2.4,
    "defect": -3.0,
    "defective": -3.0,
    "glitch": -2.0,
    "leaking": -2.2,
    "overheating": -2.8,
    "sluggish": -2.0,
    "plasticky": -1.8,
    "drain": -1.5
}

# Common consumer product aspect taxonomy
COMMON_ASPECTS = {
    "battery life": ["battery life", "battery", "charge", "charging", "runtime", "power"],
    "sound quality": ["sound quality", "audio quality", "sound", "audio", "bass", "treble", "clarity", "volume", "music"],
    "noise cancellation": ["noise cancellation", "anc", "transparency mode", "transparency", "ambient mode", "background noise"],
    "comfort & fit": ["comfort", "comfortable", "fit", "earcups", "headband", "ergonomics", "weight", "lightweight", "ear tips"],
    "build quality": ["build quality", "durability", "durable", "hinge", "sturdy", "fragile", "plastic", "build"],
    "price & value": ["price", "cost", "value", "expensive", "overpriced", "affordable", "worth the money", "worth it"],
    "performance": ["performance", "speed", "motor", "efficient", "fast", "wattage", "power"],
    "ease of use": ["ease of use", "easy to use", "setup", "controls", "buttons", "convenient", "interface"],
    "software & connectivity": ["software", "app", "bluetooth", "connection", "firmware", "pairing", "sync"],
    "cleaning & maintenance": ["cleaning", "dishwasher", "seal", "ring", "filter", "wash", "maintenance"],
    "safety & heating": ["overheating", "overheat", "heat", "heating", "burn", "spark", "temperature", "safety"]
}

class AspectExtractor:
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(CONSUMER_REVIEW_LEXICON)

        # Inverted index: keyword -> canonical aspect
        self.keyword_to_aspect: Dict[str, str] = {}
        for canonical, keywords in COMMON_ASPECTS.items():
            for kw in keywords:
                self.keyword_to_aspect[kw.lower()] = canonical

    def extract_aspect_mentions(self, text: str, rating: Optional[float] = None) -> List[Dict[str, Any]]:
        """Extract all aspect mentions and local sentiment from review text."""
        sentences = self.preprocessor.tokenize_sentences(text)
        mentions = []

        # Rating adjustment factor
        rating_boost = 0.0
        if rating is not None:
            if rating >= 4.5:
                rating_boost = 0.15
            elif rating <= 2.0:
                rating_boost = -0.15

        for sentence in sentences:
            sent_lower = sentence.lower()
            found_aspects = set()

            # Match longest keywords first to prevent partial overlaps
            for kw, canonical in sorted(self.keyword_to_aspect.items(), key=lambda x: -len(x[0])):
                pattern = r"\b" + re.escape(kw) + r"\b"
                if re.search(pattern, sent_lower):
                    found_aspects.add(canonical)

            if found_aspects:
                sentiment = self.vader.polarity_scores(sentence)
                raw_compound = sentiment["compound"]
                compound = max(-1.0, min(1.0, raw_compound + rating_boost))

                for aspect in found_aspects:
                    mentions.append({
                        "aspect": aspect,
                        "sentence": sentence.strip(),
                        "compound_score": round(compound, 3),
                        "sentiment": "positive" if compound > 0.05 else ("negative" if compound < -0.05 else "neutral")
                    })

        return mentions

    def aggregate_product_aspects(self, reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate aspect sentiments across multiple reviews for a single product."""
        aspect_data = defaultdict(lambda: {"scores": [], "positive_count": 0, "negative_count": 0, "sample_quotes": []})

        for rev in reviews:
            text = rev.get("text", "")
            rating = rev.get("rating")
            mentions = self.extract_aspect_mentions(text, rating=rating)
            for m in mentions:
                asp = m["aspect"]
                score = m["compound_score"]
                aspect_data[asp]["scores"].append(score)
                if score > 0.05:
                    aspect_data[asp]["positive_count"] += 1
                elif score < -0.05:
                    aspect_data[asp]["negative_count"] += 1

                if len(aspect_data[asp]["sample_quotes"]) < 3 and abs(score) >= 0.15:
                    aspect_data[asp]["sample_quotes"].append(m["sentence"])

        pros = []
        cons = []

        for aspect, stats in aspect_data.items():
            scores = stats["scores"]
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            total_mentions = len(scores)
            pos_ratio = (stats["positive_count"] / total_mentions) if total_mentions > 0 else 0.5

            item = {
                "aspect": aspect.title(),
                "sentiment_score": round(avg_score, 2),
                "positive_percentage": round(pos_ratio * 100, 1),
                "mention_count": total_mentions,
                "sample_quotes": stats["sample_quotes"]
            }

            if avg_score >= 0.08 or pos_ratio >= 0.60:
                pros.append(item)
            elif avg_score <= -0.08 or pos_ratio <= 0.40:
                cons.append(item)

        # Sort pros by highest sentiment and mention count
        pros.sort(key=lambda x: (x["sentiment_score"], x["mention_count"]), reverse=True)
        # Sort cons by lowest sentiment (most negative) and mention count
        cons.sort(key=lambda x: (x["sentiment_score"], -x["mention_count"]))

        return {
            "pros": pros,
            "cons": cons,
            "total_aspect_mentions": sum(len(s["scores"]) for s in aspect_data.values())
        }
