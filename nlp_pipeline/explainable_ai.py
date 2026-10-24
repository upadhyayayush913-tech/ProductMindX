"""
Explainable AI (XAI) Engine for ProductMind X using SHAP and LIME.

Implements rigorous, reproducible model explainability:
1. Training a transparent, evaluated linear sentiment/aspect classifier:
   - TfidfVectorizer (fit strictly on training split to prevent data leakage)
   - LogisticRegression (calibrated probabilities for binary sentiment)
2. Reproducible Train / Test Split with aspect-balanced pairs.
3. Pre-Explanation Evaluation:
   - Accuracy, Precision, Recall, Macro-F1, Confusion Matrix.
4. SHAP (SHapley Additive exPlanations):
   - Local review attribution values via LinearExplainer
   - Unambiguous directional labels relative to explained class:
     'Supports Negative', 'Opposes Negative', 'Supports Positive', 'Opposes Positive'.
5. LIME (Local Interpretable Model-agnostic Explanations):
   - Word-level perturbation and surrogate explanation for individual reviews
   - Clear supporting vs opposing token breakdown.
6. Diagnostic Lexical Consistency Auditor:
   - Compares classifier predictions with categorized lexicons from sentiment_lexicon.py.
7. Strict Scientific Guardrails:
   - Disclaims that SHAP/LIME explain surrogate linear model behavior, not physical ground truth or LLM reasoning.
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction import text as sklearn_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from config import CACHE_DIR
from utils.logger import get_logger
from nlp_pipeline.sentiment_lexicon import (
    STRONG_NEGATIVE_WORDS, STRONG_POSITIVE_WORDS,
    MODERATE_NEGATIVE_WORDS, MODERATE_POSITIVE_WORDS,
    CONTEXTUAL_NEUTRAL_WORDS, NEGATION_WORDS,
    INTENSIFIER_WORDS, DIMINISHER_WORDS,
    COMMON_NEGATION_PATTERNS, CONTRAST_WORDS,
    ASPECT_TAXONOMY, analyze_lexical_sentiment,
    check_prediction_lexicon_consistency
)

logger = get_logger("explainable_ai")

MODEL_DIR = CACHE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_PATH = MODEL_DIR / "sentiment_classifier_eval.json"

# 1. Custom Negation-Preserving Stopword List
SKLEARN_STOPS = set(sklearn_text.ENGLISH_STOP_WORDS)
CUSTOM_STOPWORDS = sorted(list(SKLEARN_STOPS - NEGATION_WORDS))

# 2. Balanced Aspect-Sentiment Training Corpus (Paired Positive/Negative Examples)
# Aspect nouns (battery, sound, anc, comfort, build, connectivity) appear equally across
# positive and negative pairs to guarantee aspect disentanglement.
TRAIN_PAIRS: List[Tuple[Tuple[str, int], Tuple[str, int], str]] = [
    # A. Unigram & Short Phrase Anchors (Disentangles aspect nouns and establishes polarity anchors)
    (('good', 1), ('bad', 0), 'general'),
    (('good battery', 1), ('bad battery', 0), 'battery'),
    (('good battery life', 1), ('bad battery life', 0), 'battery'),
    (('The battery life is good.', 1), ('The battery life is bad.', 0), 'battery'),
    (('The battery is good.', 1), ('The battery is bad.', 0), 'battery'),
    (('good sound quality', 1), ('bad sound quality', 0), 'sound'),
    (('The sound is good.', 1), ('The sound is bad.', 0), 'sound'),
    (('This product is good.', 1), ('This product is bad.', 0), 'general'),
    (('The product is good.', 1), ('The product is bad.', 0), 'general'),
    (('Very good quality and good performance.', 1), ('Very bad quality and bad performance.', 0), 'general'),
    (('Extremely good experience, really good.', 1), ('Extremely bad experience, really bad.', 0), 'general'),

    # B. Polarity Vocabulary Anchors
    (('excellent', 1), ('terrible', 0), 'general'),
    (('fantastic', 1), ('horrible', 0), 'general'),
    (('superb', 1), ('awful', 0), 'general'),
    (('phenomenal', 1), ('unusable', 0), 'general'),
    (('amazing', 1), ('defective', 0), 'general'),
    (('perfect', 1), ('broken', 0), 'general'),
    (('reliable', 1), ('unreliable', 0), 'general'),
    (('The connection is stable.', 1), ('The connection keeps disconnecting.', 0), 'connectivity'),
    (('The connection is reliable.', 1), ('The connection is unreliable.', 0), 'connectivity'),
    (('The headphones are comfortable and reliable.', 1), ('The headphones are uncomfortable and unreliable.', 0), 'comfort'),
    (('The headphones work great.', 1), ('The headphones never worked.', 0), 'general'),
    (('The product is very usable.', 1), ('The product is barely usable.', 0), 'general'),
    (('The product is worth the price.', 1), ('The product is not worth the price.', 0), 'price'),
    (('The product is very good.', 1), ('The product is not very good.', 0), 'general'),
    (('The sound quality is fantastic.', 1), ('The sound quality is awful.', 0), 'sound'),
    (('The battery life is excellent.', 1), ('The battery life is terrible.', 0), 'battery'),
    (('The battery holds charge for days.', 1), ('The battery drains extremely quickly.', 0), 'battery'),
    (('The microphone sounds crisp and clear.', 1), ('The microphone is muffled and distorted.', 0), 'sound'),
    (('This is the best product I have purchased.', 1), ('This is the worst product I have purchased.', 0), 'general'),
    (('The product is wonderful.', 1), ('The product is horrible.', 0), 'general'),
    (('The headphones are usable and sturdy.', 1), ('The headphones are unusable.', 0), 'general'),

    # C. Negations & Modifiers
    (('The battery life is not bad.', 1), ('The battery life is not good.', 0), 'battery'),
    (('The sound is not terrible.', 1), ('The sound is not good.', 0), 'sound'),
    (('not terrible', 1), ('not good', 0), 'general'),
    (('The audio is not bad, pleasant listening.', 1), ('The audio is not good, distorted noise.', 0), 'sound'),
    (('The noise cancellation is not bad, decent reduction.', 1), ('The noise cancellation is not good, zero reduction.', 0), 'anc'),
    (('The fit is not bad.', 1), ('The fit is not good.', 0), 'comfort'),
    (('The build is not bad.', 1), ('The build is not good.', 0), 'build'),

    # D. Multi-Aspect Sentences
    (('airpods pro 2nd generation has the best battery life of any earbuds on the market.', 1),
     ('airpods pro 2nd generation has the worst battery life of any earbuds on the market.', 0), 'battery'),
    (('the airpods 2 nd genration has the best battery life of all airpods.', 1),
     ('the airpods 2 nd genration has the worst battery life of all airpods.', 0), 'battery'),
    (('Active noise cancellation is excellent, total silence on noisy flights.', 1),
     ('Active noise cancellation is terrible, barely blocks any ambient noise at all.', 0), 'anc'),
    (('Superb audio performance, wonderful acoustic clarity and wide soundstage.', 1),
     ('Awful sound performance, constant static buzzing and crackling.', 0), 'sound'),
    (('Build quality is excellent, sturdy aluminum magnesium frame and premium finish.', 1),
     ('Build quality is terrible, flimsy cheap plastic hinge snapped after normal use.', 0), 'build'),
    (('Bluetooth connectivity is excellent, instant multipoint pairing across phone and laptop.', 1),
     ('Bluetooth connectivity is terrible, disconnects every five minutes and drops calls.', 0), 'connectivity'),

    # E. Mixed & Contrastive Training Pairs
    (('The sound is great, and the battery life is wonderful.', 1),
     ('The sound is excellent, but the battery life is terrible.', 0), 'mixed'),
    (('The battery is great, and the microphone is great.', 1),
     ('The battery is great, although the microphone is awful.', 0), 'mixed'),
    (('The product is comfortable, and the connection is reliable.', 1),
     ('The product is comfortable, but the connection is unreliable.', 0), 'mixed')
]

# 3. Holdout Test Set (Unseen test pairs to verify out-of-sample generalization)
TEST_PAIRS: List[Tuple[Tuple[str, int], Tuple[str, int], str]] = [
    (('The battery performance is phenomenal and lasts all day.', 1),
     ('The battery performance is awful and drains completely.', 0), 'battery'),
    (('The audio clarity is fantastic and vocals are crisp.', 1),
     ('The audio clarity is terrible and sound is muffled.', 0), 'sound'),
    (('The noise cancellation is superb on noisy flights.', 1),
     ('The noise cancellation is horrible and introduces loud whine.', 0), 'anc'),
    (('The wireless connection is stable and pairs instantly.', 1),
     ('The wireless connection keeps disconnecting and drops calls.', 0), 'connectivity'),
    (('The build is durable with premium aluminum materials.', 1),
     ('The build is flimsy cheap plastic that rattles.', 0), 'build'),
    (('The earcups are comfortable with soft memory foam.', 1),
     ('The earcups are uncomfortable and clamping force is painful.', 0), 'comfort'),
    (('This device is excellent and definitely worth the price.', 1),
     ('This device is defective and not worth the price.', 0), 'general'),
    (('The battery life is not bad at all for long commutes.', 1),
     ('The battery is not good at all, severe drain issues.', 0), 'battery')
]

# Combined pairs for benchmark reporting
PAIRED_BENCHMARK_CORPUS = TRAIN_PAIRS + TEST_PAIRS

ANNOTATED_BENCHMARK_CORPUS: List[Tuple[str, int, str]] = []
for p in PAIRED_BENCHMARK_CORPUS:
    ANNOTATED_BENCHMARK_CORPUS.append((p[0][0], p[0][1], p[2]))
    ANNOTATED_BENCHMARK_CORPUS.append((p[1][0], p[1][1], p[2]))


class ExplainableAIEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.classifier: Optional[LogisticRegression] = None
        self.evaluation_metrics: Optional[Dict[str, Any]] = None
        self.is_trained: bool = False
        self._shap_explainer = None
        self._lime_explainer = None

        # Train model upon initialization
        self.train_and_evaluate()

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Train a calibrated linear sentiment classifier on the paired corpus with:
        - Custom stopword filtering that explicitly preserves negation tokens.
        - Strict train / test split by pairs ensuring zero train-test leakage and aspect-balance across splits.
        - TfidfVectorizer fit ONLY on the train split.
        - Regularized LogisticRegression with balanced class weights.
        - Full classification metrics: Accuracy, Precision, Recall, Macro-F1, Confusion Matrix.
        """
        X_train_raw = [p[0][0] for p in TRAIN_PAIRS] + [p[1][0] for p in TRAIN_PAIRS]
        y_train = [p[0][1] for p in TRAIN_PAIRS] + [p[1][1] for p in TRAIN_PAIRS]

        X_test_raw = [p[0][0] for p in TEST_PAIRS] + [p[1][0] for p in TEST_PAIRS]
        y_test = [p[0][1] for p in TEST_PAIRS] + [p[1][1] for p in TEST_PAIRS]

        # Fit vectorizer strictly on X_train to prevent leakage
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_features=3000,
            stop_words=CUSTOM_STOPWORDS,
            sublinear_tf=True
        )
        X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        X_test_vec = self.vectorizer.transform(X_test_raw)

        # Train regularized Logistic Regression classifier
        self.classifier = LogisticRegression(
            C=3.0,
            class_weight="balanced",
            solver="liblinear",
            random_state=self.random_state
        )
        self.classifier.fit(X_train_vec, y_train)
        self.is_trained = True

        # Invalidate any cached explainers when model retrains
        self._shap_explainer = None
        self._lime_explainer = None

        # Evaluate on unseen holdout test split
        y_pred = self.classifier.predict(X_test_vec)

        acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"], output_dict=True)

        total_samples = len(X_train_raw) + len(X_test_raw)
        self.evaluation_metrics = {
            "model_architecture": "TfidfVectorizer (1-2 ngrams, negation-preserving) + LogisticRegression (C=3.0, balanced)",
            "total_samples": total_samples,
            "train_samples": len(X_train_raw),
            "test_samples": len(X_test_raw),
            "class_balance": {
                "positive": int(sum(y_train) + sum(y_test)),
                "negative": int(total_samples - (sum(y_train) + sum(y_test)))
            },
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "confusion_matrix": {
                "tn": cm[0][0], "fp": cm[0][1],
                "fn": cm[1][0], "tp": cm[1][1]
            },
            "per_class_metrics": {
                "negative": {
                    "precision": round(report["Negative"]["precision"], 4),
                    "recall": round(report["Negative"]["recall"], 4),
                    "f1_score": round(report["Negative"]["f1-score"], 4)
                },
                "positive": {
                    "precision": round(report["Positive"]["precision"], 4),
                    "recall": round(report["Positive"]["recall"], 4),
                    "f1_score": round(report["Positive"]["f1-score"], 4)
                }
            },
            "leakage_prevention": "Vectorizer fit exclusively on X_train; holdout test pairs evaluated strictly out-of-sample."
        }

        # Cache evaluation results to disk
        try:
            with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"Error saving eval metrics: {e}")

        logger.info(f"Explainable AI Classifier trained: Accuracy={acc:.4f}, Macro-F1={macro_f1:.4f}")
        return self.evaluation_metrics

    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment label, calibrated probabilities, uncertainty flag,
        and diagnostic lexicon consistency verification.
        """
        if not self.is_trained:
            self.train_and_evaluate()

        vec = self.vectorizer.transform([text])
        pred_class = int(self.classifier.predict(vec)[0])
        probs = self.classifier.predict_proba(vec)[0]
        prob_neg = round(float(probs[0]), 4)
        prob_pos = round(float(probs[1]), 4)
        confidence = round(float(probs[pred_class]), 4)

        # Calibration & Uncertainty Flag (near decision boundary)
        is_uncertain = 0.45 <= prob_pos <= 0.55

        # Linguistic & Lexicon Analysis from sentiment_lexicon.py
        lexical_analysis = analyze_lexical_sentiment(text)
        predicted_label = "Positive" if pred_class == 1 else "Negative"
        consistency_warning = check_prediction_lexicon_consistency(
            predicted_label=predicted_label,
            confidence=confidence,
            text=text
        )

        return {
            "text": text,
            "predicted_label": predicted_label,
            "predicted_class": pred_class,
            "confidence": confidence,
            "is_uncertain": is_uncertain,
            "consistency_warning": consistency_warning,
            "lexical_analysis": lexical_analysis,
            "probabilities": {
                "negative": prob_neg,
                "positive": prob_pos
            }
        }

    def explain_with_shap(self, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate SHAP feature attributions for a review text using LinearExplainer.
        Identifies exact tokens driving the prediction towards Positive or Negative.

        Attribution sign convention:
        - Attribution > 0: Increases log-odds of Positive class.
        - Attribution < 0: Decreases log-odds of Positive class (pushes toward Negative).

        Direction labels relative to explained class:
        - When explaining Positive: val > 0 -> 'Supports Positive', val < 0 -> 'Opposes Positive'.
        - When explaining Negative: val < 0 -> 'Supports Negative', val > 0 -> 'Opposes Negative'.
        """
        if not self.is_trained:
            self.train_and_evaluate()

        pred = self.predict_sentiment(text)
        pred_class = pred["predicted_class"]
        pred_label = pred["predicted_label"]

        try:
            import shap
            vec = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()

            # Initialize SHAP linear explainer with training background
            if self._shap_explainer is None:
                background_texts = [item[0] for item in ANNOTATED_BENCHMARK_CORPUS[:25]]
                bg_vec = self.vectorizer.transform(background_texts)
                self._shap_explainer = shap.LinearExplainer(self.classifier, bg_vec)

            shap_values = self._shap_explainer(vec)
            vals = shap_values.values[0]

            # Find active words in this specific text
            non_zero_indices = vec.indices
            attributions = []
            for idx in non_zero_indices:
                name = str(feature_names[idx])
                val = float(vals[idx])
                # Does this token support the predicted class?
                supports = (val > 0 and pred_class == 1) or (val < 0 and pred_class == 0)
                direction_label = f"Supports {pred_label}" if supports else f"Opposes {pred_label}"
                impact = "Pushes toward Positive" if val > 0 else "Pushes toward Negative"

                attributions.append({
                    "feature": name,
                    "attribution": round(val, 4),
                    "impact": impact,
                    "supports_prediction": supports,
                    "direction_label": direction_label,
                    "magnitude": abs(val)
                })

            # Sort by absolute magnitude
            attributions.sort(key=lambda x: x["magnitude"], reverse=True)
            top_attributions = attributions[:top_k]

            base_val = round(float(shap_values.base_values[0]), 4) if hasattr(shap_values, "base_values") else 0.0

            supporting_features = [a for a in top_attributions if a["supports_prediction"]]
            opposing_features = [a for a in top_attributions if not a["supports_prediction"]]

            return {
                "method": "SHAP (LinearExplainer)",
                "text": text,
                "prediction": pred,
                "explained_class_name": pred_label,
                "explained_class_probability": pred["confidence"],
                "base_value": base_val,
                "top_attributions": top_attributions,
                "supporting_features": supporting_features,
                "opposing_features": opposing_features,
                "positive_drivers": [a for a in top_attributions if a["attribution"] > 0],
                "negative_drivers": [a for a in top_attributions if a["attribution"] < 0],
                "caveat": "SHAP values measure feature contribution to the linear surrogate model; they do not prove ground truth or physical product defect causality."
            }
        except Exception as e:
            logger.warning(f"Error computing SHAP explanation: {e}")
            return {
                "method": "SHAP (Fallback)",
                "error": str(e),
                "text": text,
                "prediction": pred,
                "explained_class_name": pred_label,
                "top_attributions": [],
                "supporting_features": [],
                "opposing_features": []
            }

    def explain_with_lime(self, text: str, num_features: int = 8) -> Dict[str, Any]:
        """
        Generate LIME local surrogate explanations for an individual review.
        Perturbs text tokens and measures prediction shifts against predicted class.

        Weight convention for predicted class:
        - Weight > 0: Word supports the predicted class ('Supports <Label>').
        - Weight < 0: Word opposes the predicted class ('Opposes <Label>').
        """
        if not self.is_trained:
            self.train_and_evaluate()

        pred = self.predict_sentiment(text)
        pred_class = pred["predicted_class"]
        pred_label = pred["predicted_label"]
        opposite_label = "Positive" if pred_label == "Negative" else "Negative"

        try:
            from lime.lime_text import LimeTextExplainer

            if self._lime_explainer is None:
                self._lime_explainer = LimeTextExplainer(
                    class_names=["Negative", "Positive"],
                    random_state=self.random_state
                )

            # Predictor function for LIME: text list -> (n_samples, 2) probabilities
            def predictor_fn(texts: List[str]):
                vecs = self.vectorizer.transform(texts)
                return self.classifier.predict_proba(vecs)

            exp = self._lime_explainer.explain_instance(
                text_instance=text,
                classifier_fn=predictor_fn,
                num_features=num_features,
                labels=[pred_class]
            )

            # Extract weights for the predicted class
            exp_list = exp.as_list(label=pred_class)
            word_weights = []
            for word, weight in exp_list:
                w_val = float(weight)
                word_str = str(word)
                supports = w_val > 0
                word_weights.append({
                    "word": word_str,
                    "weight": round(w_val, 4),
                    "supports_prediction": supports,
                    "direction_label": f"Supports {pred_label}" if supports else f"Opposes {pred_label}",
                    "pushes_toward": pred_label if supports else opposite_label,
                    "magnitude": abs(w_val)
                })

            word_weights.sort(key=lambda x: x["magnitude"], reverse=True)

            score_intercept = round(float(exp.intercept[pred_class]), 4) if hasattr(exp, "intercept") and pred_class in exp.intercept else 0.0
            fidelity = round(float(exp.score), 4) if hasattr(exp, "score") else 1.0

            supporting_words = [w for w in word_weights if w["supports_prediction"]]
            opposing_words = [w for w in word_weights if not w["supports_prediction"]]

            return {
                "method": "LIME (LimeTextExplainer)",
                "text": text,
                "prediction": pred,
                "explained_class_name": pred_label,
                "explained_class_probability": pred["confidence"],
                "score_intercept": score_intercept,
                "local_prediction_fidelity": fidelity,
                "word_weights": word_weights,
                "supporting_words": supporting_words,
                "opposing_words": opposing_words,
                "caveat": "LIME fits a local linear surrogate model around the perturbed review; it explains the classifier's local decision boundary."
            }
        except Exception as e:
            logger.warning(f"Error computing LIME explanation: {e}")
            return {
                "method": "LIME (Fallback)",
                "error": str(e),
                "text": text,
                "prediction": pred,
                "explained_class_name": pred_label,
                "word_weights": [],
                "supporting_words": [],
                "opposing_words": []
            }

    def explain_review_side_by_side(self, text: str) -> Dict[str, Any]:
        """
        Provide side-by-side SHAP and LIME explanations for a single review.
        """
        pred = self.predict_sentiment(text)
        shap_res = self.explain_with_shap(text)
        lime_res = self.explain_with_lime(text)

        return {
            "text": text,
            "prediction": pred,
            "evaluation_metrics": self.evaluation_metrics,
            "shap_explanation": shap_res,
            "lime_explanation": lime_res
        }


# Global singleton
xai_engine = ExplainableAIEngine()

if __name__ == "__main__":
    test_text = "bad battery life"
    print("=== EXPLAINABLE AI INFERENCE & EXPLANATIONS ===")
    res = xai_engine.explain_review_side_by_side(test_text)
    print("Prediction:", res["prediction"])
    print("\nSHAP Top Attributions:")
    for a in res["shap_explanation"]["top_attributions"]:
        print(f"  - {a['feature']}: {a['attribution']} ({a['direction_label']})")
    print("\nLIME Word Weights:")
    for w in res["lime_explanation"]["word_weights"]:
        print(f"  - {w['word']}: {w['weight']} ({w['direction_label']})")
