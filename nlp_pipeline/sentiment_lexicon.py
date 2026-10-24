"""
Sentiment Lexicon Module for ProductMind X.

Provides transparent, categorized linguistic sentiment dictionaries,
negation-handling rules, contrast indicators, and aspect taxonomies:
1. Strong / Moderate Positive and Negative lexicons.
2. Contextual neutral and aspect-specific terms.
3. Negation modifiers, intensifiers, and diminishers.
4. Clause-level contrast and mixed-sentiment handling.
5. Lexical diagnostics and classifier consistency auditing.
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any
from sklearn.feature_extraction import text as sklearn_text

# ============================================================================
# Stopwords (Negation-Preserving)
# ============================================================================
SKLEARN_STOPS = set(sklearn_text.ENGLISH_STOP_WORDS)
CUSTOM_STOPWORDS = sorted(list(SKLEARN_STOPS - {"not", "no", "never", "neither", "nor", "without", "hardly", "barely", "scarcely", "cannot"}))

# ============================================================================
# A. Strong Negative Vocabulary
# ============================================================================
STRONG_NEGATIVE_WORDS: Set[str] = {
    "bad", "worst", "awful", "terrible", "horrible", "horrendous", "dreadful",
    "atrocious", "abysmal", "appalling", "pathetic", "disappointing",
    "unsatisfactory", "unacceptable", "defective", "broken", "damaged",
    "faulty", "unreliable", "unusable", "worthless", "garbage", "trash",
    "junk", "rubbish", "crap", "frustrating", "infuriating", "annoying",
    "irritating", "poor", "inferior", "subpar", "inadequate", "insufficient",
    "noisy", "uncomfortable", "fragile", "flimsy", "sluggish", "laggy",
    "overpriced", "wasteful", "useless", "malfunctioning", "fails", "failed",
    "failing", "failure", "stopped", "died", "disconnects", "drains",
    "draining", "overheating", "overheats", "crashing", "crashes", "distorted",
    "muffled", "scratchy", "crackling", "intermittent", "short-lived", "weak",
    "inconsistent", "unstable", "hate", "hated", "useless", "bricked"
}

# ============================================================================
# B. Strong Positive Vocabulary
# ============================================================================
STRONG_POSITIVE_WORDS: Set[str] = {
    "excellent", "exceptional", "outstanding", "superb", "fantastic",
    "phenomenal", "amazing", "incredible", "wonderful", "brilliant",
    "impressive", "remarkable", "great", "good", "best", "perfect",
    "delightful", "reliable", "durable", "comfortable", "sturdy", "premium",
    "smooth", "responsive", "fast", "crisp", "clear", "balanced", "powerful",
    "flawless", "dependable", "convenient", "efficient", "worthwhile",
    "affordable", "valuable", "recommended", "satisfied", "exceeded",
    "exceeds", "love", "loved", "long-lasting", "stable", "seamless",
    "effective", "accurate"
}

# ============================================================================
# C. Moderate Sentiment & Context-Dependent Vocabulary
# ============================================================================
MODERATE_NEGATIVE_WORDS: Set[str] = {
    "mediocre", "limited", "inconvenient", "slightly uncomfortable",
    "somewhat noisy", "clunky", "finicky", "lukewarm", "plain", "slow"
}

MODERATE_POSITIVE_WORDS: Set[str] = {
    "decent", "satisfactory", "acceptable", "useful", "reasonable",
    "fairly good", "pleasant", "adequate", "solid", "fair", "neat", "fine"
}

# Inherently neutral terms whose polarity is strictly context- and aspect-dependent
CONTEXTUAL_NEUTRAL_WORDS: Set[str] = {
    "average", "expensive", "large", "small", "heavy", "light", "standard",
    "normal", "basic", "compact", "expected"
}

# ============================================================================
# D. Negation & Polarity Modifiers
# ============================================================================
NEGATION_WORDS: Set[str] = {
    "not", "no", "never", "neither", "nor", "without", "hardly", "barely",
    "scarcely", "cannot", "cant", "can't", "wont", "won't", "dont", "don't",
    "doesnt", "doesn't", "didnt", "didn't", "isnt", "isn't", "arent", "aren't"
}

INTENSIFIER_WORDS: Set[str] = {
    "very", "extremely", "incredibly", "highly", "really", "absolutely",
    "completely", "totally", "exceptionally", "too"
}

DIMINISHER_WORDS: Set[str] = {
    "slightly", "somewhat", "mildly", "fairly", "quite", "less", "almost",
    "marginally", "a bit", "partially"
}

# Common multi-word negation phrases
COMMON_NEGATION_PATTERNS = [
    (r"\bnot\s+good\b", -0.7),
    (r"\bnot\s+bad\b", +0.65),
    (r"\bnot\s+very\s+good\b", -0.6),
    (r"\bnever\s+worked\b", -0.9),
    (r"\bbarely\s+usable\b", -0.8),
    (r"\bno\s+longer\s+works?\b", -0.9),
    (r"\bnot\s+worth\b", -0.85),
    (r"\bnot\s+as\s+good\b", -0.5),
    (r"\bfar\s+from\s+perfect\b", -0.6),
    (r"\bnot\s+terrible\b", +0.5),
    (r"\bwithout\s+(any\s+)?issues?\b", +0.8),
    (r"\bwithout\s+(any\s+)?problems?\b", +0.8),
    (r"\bwithout\s+(any\s+)?defects?\b", +0.8)
]

# ============================================================================
# E. Contrast & Clause Boundaries
# ============================================================================
CONTRAST_WORDS: Set[str] = {
    "but", "however", "although", "though", "whereas", "yet", "despite",
    "except", "while", "nevertheless"
}

# ============================================================================
# F. Product Aspect Taxonomies
# ============================================================================
ASPECT_TAXONOMY: Dict[str, Set[str]] = {
    "battery life": {
        "battery", "battery life", "runtime", "charge", "charging", "charger",
        "power", "drains", "drain", "cell", "mah", "charging case", "usb-c"
    },
    "sound quality": {
        "sound", "audio", "sound quality", "music", "bass", "treble", "mids",
        "vocals", "clarity", "acoustic", "volume", "distort", "distortion",
        "speaker", "frequency", "tinny", "muffled"
    },
    "noise cancellation": {
        "anc", "noise cancellation", "noise cancelling", "isolation", "ambient",
        "transparency", "transparency mode", "cabin pressure", "whine"
    },
    "comfort & fit": {
        "comfort", "fit", "earcups", "ear cushions", "ear tips", "headband",
        "clamping", "pressure", "weight", "ergonomic", "fatigue", "aches"
    },
    "build quality": {
        "build", "build quality", "hinge", "hinges", "plastic", "aluminum",
        "magnesium", "durability", "durable", "sturdy", "flimsy", "rattles",
        "cracked", "snapped", "materials"
    },
    "software & connectivity": {
        "bluetooth", "wireless", "pairing", "connection", "connects", "app",
        "firmware", "latency", "multipoint", "disconnects", "stuttering"
    },
    "price & value": {
        "price", "value", "cost", "expensive", "affordable", "worth", "money",
        "warranty", "customer service"
    }
}


def analyze_lexical_sentiment(text: str) -> Dict[str, Any]:
    """
    Perform transparent lexical parsing across sentiment words, negations,
    intensifiers, clause-level contrasts, and product aspects.
    """
    text_lower = text.lower().strip()
    words = re.findall(r"\b[a-z\'-]+\b", text_lower)

    detected_strong_pos = [w for w in words if w in STRONG_POSITIVE_WORDS]
    detected_strong_neg = [w for w in words if w in STRONG_NEGATIVE_WORDS]
    detected_mod_pos = [w for w in words if w in MODERATE_POSITIVE_WORDS]
    detected_mod_neg = [w for w in words if w in MODERATE_NEGATIVE_WORDS]
    detected_negations = [w for w in words if w in NEGATION_WORDS]
    detected_contrasts = [w for w in words if w in CONTRAST_WORDS]

    # Detect product aspects
    detected_aspects = []
    for aspect_name, tokens in ASPECT_TAXONOMY.items():
        if any(t in text_lower for t in tokens):
            detected_aspects.append(aspect_name)

    # Calculate token-level polarity with window-based negation
    score = 0.0
    negation_active = False
    negation_window = 0

    for idx, w in enumerate(words):
        if w in NEGATION_WORDS:
            negation_active = True
            negation_window = 3
            continue

        weight = 0.0
        if w in STRONG_POSITIVE_WORDS:
            weight = 1.0
        elif w in STRONG_NEGATIVE_WORDS:
            weight = -1.0
        elif w in MODERATE_POSITIVE_WORDS:
            weight = 0.5
        elif w in MODERATE_NEGATIVE_WORDS:
            weight = -0.5

        if weight != 0.0:
            # Check preceding intensifier
            if idx > 0 and words[idx - 1] in INTENSIFIER_WORDS:
                weight *= 1.3
            elif idx > 0 and words[idx - 1] in DIMINISHER_WORDS:
                weight *= 0.7

            if negation_active:
                # Negating a positive word makes it moderately negative
                # Negating a negative word makes it moderately positive ("not bad" -> +0.6)
                weight = -0.7 * weight

            score += weight

        if negation_active:
            negation_window -= 1
            if negation_window <= 0:
                negation_active = False

    # Check explicit multi-word negation patterns
    for pat, pat_weight in COMMON_NEGATION_PATTERNS:
        if re.search(pat, text_lower):
            score += pat_weight

    # Determine verdict
    if score > 0.35:
        verdict = "Positive"
    elif score < -0.35:
        verdict = "Negative"
    else:
        verdict = "Neutral"

    return {
        "text": text,
        "score": round(score, 3),
        "verdict": verdict,
        "strong_positive_tokens": list(set(detected_strong_pos)),
        "strong_negative_tokens": list(set(detected_strong_neg)),
        "moderate_positive_tokens": list(set(detected_mod_pos)),
        "moderate_negative_tokens": list(set(detected_mod_neg)),
        "negation_tokens": list(set(detected_negations)),
        "contrast_tokens": list(set(detected_contrasts)),
        "detected_aspects": detected_aspects,
        "has_contrast": len(detected_contrasts) > 0
    }


def check_prediction_lexicon_consistency(
    predicted_label: str,
    confidence: float,
    text: str
) -> Optional[str]:
    """
    Diagnostic consistency auditor. Compares model prediction with lexical cues
    and returns a transparent warning if a stark discrepancy exists.
    Does NOT modify the model's prediction.
    """
    analysis = analyze_lexical_sentiment(text)
    strong_pos = analysis["strong_positive_tokens"]
    strong_neg = analysis["strong_negative_tokens"]
    has_contrast = analysis["has_contrast"]

    # If sentence has contrast clauses ("Sound is good, but battery is bad"),
    # mixed sentiment is expected and not necessarily an inconsistency.
    if has_contrast:
        if strong_pos and strong_neg:
            return "Mixed sentiment detected across contrast clauses (e.g. 'but', 'however'). Aspect-level breakdown advised."

    # Strong negative markers present with no positive markers, but model predicted Positive
    if strong_neg and not strong_pos and predicted_label == "Positive":
        return f"Lexical Discrepancy: Strong negative sentiment tokens {strong_neg} detected, but classifier predicted Positive."

    # Strong positive markers present with no negative markers, but model predicted Negative
    if strong_pos and not strong_neg and predicted_label == "Negative":
        return f"Lexical Discrepancy: Strong positive sentiment tokens {strong_pos} detected, but classifier predicted Negative."

    return None
