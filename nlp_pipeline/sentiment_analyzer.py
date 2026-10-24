"""
Sentiment Analysis Module.
Dual approach combining:
- VADER (Valence Aware Dictionary and sEntiment Reasoner)
- Transformer (distilbert-base-uncased-finetuned-sst-2) with fast fallback
- Star-rating calibration
"""

from typing import Dict, Any, List, Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import TRANSFORMER_SENTIMENT_MODEL
from utils.logger import get_logger

logger = get_logger("sentiment_analyzer")

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
    "overheating": -2.8
}

class SentimentAnalyzer:
    def __init__(self, use_transformer: bool = False):
        self.vader = SentimentIntensityAnalyzer()
        self.vader.lexicon.update(CONSUMER_REVIEW_LEXICON)
        self.use_transformer = use_transformer
        self.transformer_pipeline = None

        if self.use_transformer:
            self._load_transformer()

    def _load_transformer(self):
        try:
            from transformers import pipeline
            logger.info(f"Loading transformer sentiment model: {TRANSFORMER_SENTIMENT_MODEL}")
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model=TRANSFORMER_SENTIMENT_MODEL,
                truncation=True,
                max_length=512,
                device=-1 # CPU
            )
        except Exception as e:
            logger.warning(f"Could not load transformer pipeline ({e}). Defaulting to VADER.")
            self.transformer_pipeline = None

    def analyze_text(self, text: str, rating: Optional[float] = None) -> Dict[str, Any]:
        """Analyze sentiment of review text and optional star rating."""
        if not text or not text.strip():
            return {
                "vader_compound": 0.0,
                "transformer_label": "NEUTRAL",
                "transformer_score": 0.5,
                "final_sentiment": "neutral",
                "confidence": 0.5
            }

        # 1. VADER analysis
        vader_scores = self.vader.polarity_scores(text)
        compound = vader_scores["compound"]

        # 2. Transformer analysis (if enabled & loaded)
        transformer_label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
        transformer_score = abs(compound)

        if self.transformer_pipeline:
            try:
                res = self.transformer_pipeline(text[:512])[0]
                transformer_label = res["label"].upper()
                transformer_score = float(res["score"])
            except Exception as e:
                logger.debug(f"Transformer inference error: {e}")

        # 3. Rating calibration (if rating is given)
        # 1-2 stars = strong negative prior, 4-5 stars = strong positive prior
        rating_boost = 0.0
        if rating is not None:
            if rating <= 2.0:
                rating_boost = -0.3
            elif rating >= 4.0:
                rating_boost = 0.3

        calibrated_score = compound + rating_boost
        calibrated_score = max(-1.0, min(1.0, calibrated_score))

        # Determine final label and confidence
        if calibrated_score >= 0.15:
            final_sentiment = "positive"
            confidence = round(min(1.0, 0.5 + (calibrated_score / 2.0)), 2)
        elif calibrated_score <= -0.15:
            final_sentiment = "negative"
            confidence = round(min(1.0, 0.5 + (abs(calibrated_score) / 2.0)), 2)
        else:
            final_sentiment = "neutral"
            confidence = 0.65

        return {
            "vader_compound": round(compound, 3),
            "vader_pos": round(vader_scores["pos"], 3),
            "vader_neg": round(vader_scores["neg"], 3),
            "vader_neu": round(vader_scores["neu"], 3),
            "transformer_label": transformer_label,
            "transformer_score": round(transformer_score, 3),
            "final_sentiment": final_sentiment,
            "calibrated_score": round(calibrated_score, 3),
            "confidence": confidence
        }

    def batch_analyze(self, texts: List[str], ratings: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Batch analyze a collection of texts."""
        results = []
        for i, text in enumerate(texts):
            r = ratings[i] if ratings and i < len(ratings) else None
            results.append(self.analyze_text(text, rating=r))
        return results
