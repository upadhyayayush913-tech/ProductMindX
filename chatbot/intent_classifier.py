"""
Intent Classifier for ProductMind X Cognitive Consumer Intelligence System.

Supports 10 Canonical, Testable Analytical Intents:
1.  general_qa: General review question -> retrieval + evidence-grounded synthesis.
2.  sentiment: Sentiment question -> sentiment prediction / aggregation + evidence.
3.  aspect: Aspect question -> aspect extraction / analysis + aspect-linked reviews.
4.  emerging_issue: Emerging issue question -> temporal comparison + complaint rates + statistical hypothesis testing.
5.  product_comparison: Product comparison -> product identity resolution + matched scope + aspect comparison.
6.  explanation: Explanation request -> inspect relevant model output or surrogate execution trace (SHAP/LIME).
7.  counter_evidence: Challenge to an answer -> retrieve counter-evidence + re-evaluate claims & surface contradictions.
8.  retrieval_comparison: Retrieval-method question -> compare actual BM25, dense ChromaDB, and hybrid results.
9.  methodology: Methodology question -> explain implemented algorithms and configuration from source code.
10. insufficient_evidence: Unknown product / insufficient evidence -> explain gap and suggest narrower query.

Also preserves backwards-compatibility aliases for legacy intents (pros_cons, issues, comparison, general_chat).
"""

import re
from typing import Dict, Any, List, Optional
from utils.logger import get_logger

logger = get_logger("intent_classifier")

CANONICAL_INTENTS = [
    "general_qa",
    "sentiment",
    "aspect",
    "emerging_issue",
    "product_comparison",
    "explanation",
    "counter_evidence",
    "retrieval_comparison",
    "methodology",
    "insufficient_evidence"
]

INTENT_DEFINITIONS = {
    # 1. Explanation request (SHAP / LIME / model inspection)
    "explanation": [
        r"\b(?:why\s+(?:this\s+)?prediction|explain\s+(?:the\s+)?(?:prediction|model|classifier)|why\s+was\s+(?:this|it)\s+classified)\b",
        r"\b(?:shap|lime|feature\s+attribution|token\s+weights?|surrogate\s+model|decision\s+boundary|why\s+positive|why\s+negative)\b",
        r"\b(?:why\s+did\s+the\s+model\s+predict|how\s+did\s+the\s+classifier\s+decide|why\s+this\s+answer)\b"
    ],

    # 2. Counter-evidence & Answer Challenge
    "counter_evidence": [
        r"\b(?:counter[- ]evidence|contradictory\s+evidence|contrary\s+reviews?|opposing\s+evidence|are\s+there\s+(?:any\s+)?contradictions?)\b",
        r"\b(?:challenge\s+(?:this\s+)?(?:answer|claim)|are\s+you\s+sure|disagree|show\s+(?:me\s+)?opposing\s+views|counter\s+claims?)\b",
        r"\b(?:is\s+there\s+any\s+counter\s+evidence|re[- ]evaluate\s+(?:claims?|answer)|contrasting\s+opinions?)\b"
    ],

    # 3. Retrieval method comparison
    "retrieval_comparison": [
        r"\b(?:compare\s+retrieval|retrieval\s+comparison|bm25\s+vs\s+dense|dense\s+vs\s+bm25|hybrid\s+vs\s+dense|retrieval\s+method)\b",
        r"\b(?:how\s+does\s+bm25\s+compare|lexical\s+vs\s+semantic|dense\s+chromadb\s+vs\s+bm25|vector\s+search\s+vs\s+lexical)\b"
    ],

    # 4. Product Comparison & Recommendation
    "product_comparison": [
        r"\b(?:compare|comparison|versus|vs\.?|difference\s+between|better\s+than|which\s+(?:one\s+)?(?:is\s+)?better|which\s+(?:one\s+)?is\s+best)\b",
        r"\b(?:which\s+(?:one\s+)?should\s+i\s+buy|what\s+should\s+i\s+buy|should\s+i\s+buy|recommend\s+(?:me|one)?|recommendation|buying\s+advice)\b"
    ],

    # 5. Emerging Issues & Temporal Complaints
    "emerging_issue": [
        r"\b(?:defect|defects|flaw|flaws|break|broke|broken|complaint|complaints|fail|failure|glitch|bug|trouble|downside|risk)\b",
        r"\b(?:getting\s+worse|getting\s+better|increasing\s+over\s+time|trend|timeline|emerging\s+issue|spiking|rising\s+complaints?)\b",
        r"\b(?:did\s+complaints\s+increase|complaint\s+rate|temporal\s+comparison|statistically\s+supported\s+increase|two[- ]proportion)\b",
        r"\b(?:common\s+defects?|failure\s+rate|what\s+goes\s+wrong|stops?\s+working)\b"
    ],

    # 6. Aspect Analysis & Pros/Cons
    "aspect": [
        r"\b(?:battery\s+life|sound\s+quality|noise\s+cancellation|comfort|build\s+quality|connectivity|bluetooth|microphone|anc)\b",
        r"\b(?:aspect|aspects|aspect[- ]based|absa|pros?\s+and\s+cons?|advantages?\s+and\s+disadvantages?|strengths?\s+and\s+weaknesses?)\b",
        r"\b(?:what\s+do\s+people\s+like|what\s+do\s+people\s+hate|best\s+aspects?|worst\s+aspects?)\b"
    ],

    # 7. Sentiment Questions
    "sentiment": [
        r"\b(?:sentiment|overall\s+sentiment|positive\s+or\s+negative|is\s+it\s+(?:liked|disliked|popular|well[- ]received))\b",
        r"\b(?:how\s+positive|how\s+negative|sentiment\s+score|sentiment\s+distribution|polarity|sentiment\s+prediction)\b"
    ],

    # 8. Methodology & Algorithmic Explanation
    "methodology": [
        r"\b(?:how\s+(?:is|was)\s+this\s+calculated|explain\s+the\s+methodology|what\s+formula|scoring\s+algorithm|calculation\s+definition)\b",
        r"\b(?:how\s+does\s+the\s+algorithm\s+work|margin\s+of\s+error|two[- ]proportion\s+z[- ]test|fisher\s+exact|benjamini[- ]hochberg|fdr)\b"
    ],

    # 9. General QA & Product Lookup
    "general_qa": [
        r"\b(?:tell\s+me\s+about|what\s+is|overview|details|features|specs|specifications|price|how\s+much\s+is|description)\b",
        r"\b(?:what\s+customers?\s+say|summary\s+of\s+reviews|customer\s+feedback|user\s+experience|general\s+opinion)\b",
        r"^\s*(?:hi|hello|hey|greetings|help|what\s+can\s+you\s+do|capabilities)\b"
    ],

    # 10. Insufficient Evidence / Unknown Product
    "insufficient_evidence": [
        r"\b(?:unindexed|unsupported|insufficient\s+evidence|unknown\s+product|non[- ]existent|not\s+in\s+(?:the\s+)?database|unverified\s+product|missing\s+evidence)\b",
        r"\b(?:data\s+gap|unsupported\s+entity|not\s+covered|outside\s+(?:the\s+)?corpus)\b"
    ]
}

# Legacy intent backward compatibility map
LEGACY_INTENT_MAP = {
    "pros_cons": "aspect",
    "issues": "emerging_issue",
    "comparison": "product_comparison",
    "comparison_recommendation": "product_comparison",
    "temporal_trend": "emerging_issue",
    "methodology_explanation": "methodology",
    "evidence_lookup": "counter_evidence",
    "product_lookup": "general_qa",
    "general_capability": "general_qa",
    "general_chat": "general_qa",
    "ambiguous_unsupported": "insufficient_evidence",
    "aspect_analysis": "aspect"
}


class IntentClassifier:
    def __init__(self):
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_DEFINITIONS.items()
        }

    def classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Classify user query into one of the 10 canonical intents.
        Deterministic regex scoring with priority hierarchy.
        """
        q = query.strip()
        lower_q = q.lower()
        matched_scores: Dict[str, int] = {}

        for intent, patterns in self.compiled_patterns.items():
            score = 0
            for pat in patterns:
                if pat.search(lower_q):
                    score += 1
            if score > 0:
                matched_scores[intent] = score

        if not matched_scores:
            # Fallback for short ungrounded queries
            words = q.split()
            if len(words) <= 2 and not any(w in lower_q for w in ["hi", "hello", "help"]):
                primary_intent = "insufficient_evidence"
                confidence = 0.50
            else:
                primary_intent = "general_qa"
                confidence = 0.65
        else:
            # Priority rank to break ties deterministically
            priority_rank = [
                "explanation",           # Explaining model or prediction takes highest precedence
                "counter_evidence",      # Challenging claims
                "retrieval_comparison",  # Algorithmic comparisons
                "methodology",           # Mathematical definitions
                "product_comparison",    # Head-to-head comparisons
                "emerging_issue",        # Complaints and temporal defect analysis
                "aspect",                # Aspect-level inquiry
                "sentiment",             # Sentiment breakdown
                "general_qa",            # Default general inquiry
                "insufficient_evidence"
            ]

            sorted_intents = sorted(
                matched_scores.keys(),
                key=lambda x: (matched_scores[x], -priority_rank.index(x) if x in priority_rank else 0),
                reverse=True
            )
            primary_intent = sorted_intents[0]
            confidence = 0.90

        return {
            "intent": primary_intent,
            "confidence": confidence,
            "all_scores": matched_scores,
            "canonical_intent": primary_intent,
            "is_canonical": primary_intent in CANONICAL_INTENTS
        }

    def map_legacy_intent(self, legacy_intent: str) -> str:
        """Map legacy intent names to the 10 canonical intents."""
        return LEGACY_INTENT_MAP.get(legacy_intent, legacy_intent)
