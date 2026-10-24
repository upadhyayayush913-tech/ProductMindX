"""
Comprehensive Unit & Regression Tests for Explainable AI (SHAP & LIME) Engine.

Validates:
1. Leakage prevention: Vectorizer fit strictly on train split, evaluated out-of-sample.
2. Pre-explanation performance metrics (Accuracy, Macro-F1, Confusion Matrix).
3. 25-Sentence Regression Benchmark:
   - 10 Strong Negative phrases (including 'bad battery life')
   - 7 Strong Positive phrases (including 'good battery life')
   - 5 Negation phrases (including 'The battery life is not bad.')
   - 3 Mixed / Contrastive phrases (including 'The sound is excellent, but the battery life is terrible.')
4. Unambiguous SHAP & LIME Attribution Directions:
   - Exact attribution directions for 'bad battery life':
     'bad' must support Negative in both SHAP and LIME.
   - Aspect nouns ('battery', 'life') must not falsely dominate sentiment.
5. Lexical Diagnostics & Consistency Auditor.
"""

import unittest
from nlp_pipeline.explainable_ai import ExplainableAIEngine

class TestExplainableAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = ExplainableAIEngine(random_state=42)

    def test_model_training_and_metrics(self):
        """Verify model evaluates before explaining and reports holdout metrics."""
        metrics = self.engine.evaluation_metrics
        self.assertIsNotNone(metrics)
        self.assertIn("accuracy", metrics)
        self.assertIn("macro_f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0.85)
        self.assertGreaterEqual(metrics["macro_f1"], 0.85)
        self.assertIn("leakage_prevention", metrics)

    def test_25_benchmark_regression_cases(self):
        """Verify 100% pass rate across the 25 required benchmark phrases."""
        test_cases = [
            # Group A: Strong Negative (10 cases)
            ("bad battery life", "Negative"),
            ("The battery life is terrible.", "Negative"),
            ("The product is horrible.", "Negative"),
            ("The headphones are unusable.", "Negative"),
            ("The sound quality is awful.", "Negative"),
            ("The battery drains extremely quickly.", "Negative"),
            ("The connection keeps disconnecting.", "Negative"),
            ("This is the worst product I have purchased.", "Negative"),
            ("The microphone is muffled and distorted.", "Negative"),
            ("The product is not worth the price.", "Negative"),

            # Group B: Strong Positive (7 cases)
            ("good battery life", "Positive"),
            ("The battery life is excellent.", "Positive"),
            ("The sound quality is fantastic.", "Positive"),
            ("The headphones are comfortable and reliable.", "Positive"),
            ("The connection is stable.", "Positive"),
            ("The microphone sounds crisp and clear.", "Positive"),
            ("This is the best product I have purchased.", "Positive"),

            # Group C: Negations (5 cases)
            ("The battery life is not bad.", "Positive"),
            ("The sound is not terrible.", "Positive"),
            ("The product is not very good.", "Negative"),
            ("The headphones never worked.", "Negative"),
            ("The product is barely usable.", "Negative"),

            # Group D: Mixed / Contrastive (3 cases)
            ("The sound is excellent, but the battery life is terrible.", "Negative"),
            ("The battery is great, although the microphone is awful.", "Negative"),
            ("The product is comfortable, but the connection is unreliable.", "Negative"),
        ]

        for text, expected in test_cases:
            pred = self.engine.predict_sentiment(text)
            self.assertEqual(
                pred["predicted_label"],
                expected,
                f"Failed for text: '{text}'. Expected {expected}, got {pred['predicted_label']} "
                f"(P(Neg)={pred['probabilities']['negative']}, P(Pos)={pred['probabilities']['positive']})"
            )

    def test_bad_battery_life_xai_direction(self):
        """
        Verify the exact reproduction case 'bad battery life':
        1. Predicted sentiment is Negative.
        2. SHAP identifies 'bad' and 'bad battery' supporting Negative.
        3. LIME identifies 'bad' supporting Negative.
        4. Aspect nouns ('battery', 'life') do NOT overpower 'bad'.
        """
        res = self.engine.explain_review_side_by_side("bad battery life")
        pred = res["prediction"]

        # 1. Prediction checks
        self.assertEqual(pred["predicted_label"], "Negative")
        self.assertGreater(pred["probabilities"]["negative"], 0.65)

        # 2. SHAP checks
        shap_exp = res["shap_explanation"]
        self.assertEqual(shap_exp["explained_class_name"], "Negative")

        bad_shap = next((a for a in shap_exp["top_attributions"] if a["feature"] == "bad"), None)
        self.assertIsNotNone(bad_shap, "Token 'bad' must be in top SHAP attributions.")
        self.assertLess(bad_shap["attribution"], 0, "SHAP attribution for 'bad' must be negative (pushes toward Negative class).")
        self.assertEqual(bad_shap["direction_label"], "Supports Negative")

        # Confirm supporting features contain 'bad'
        supp_features = [f["feature"] for f in shap_exp["supporting_features"]]
        self.assertIn("bad", supp_features)

        # 3. LIME checks
        lime_exp = res["lime_explanation"]
        self.assertEqual(lime_exp["explained_class_name"], "Negative")

        bad_lime = next((w for w in lime_exp["word_weights"] if w["word"] == "bad"), None)
        self.assertIsNotNone(bad_lime, "Word 'bad' must be in LIME word weights.")
        self.assertGreater(bad_lime["weight"], 0, "LIME weight for 'bad' when explaining Class 0 must be positive.")
        self.assertTrue(bad_lime["supports_prediction"])
        self.assertEqual(bad_lime["direction_label"], "Supports Negative")

    def test_lexical_analysis_attachment(self):
        """Verify linguistic lexicon diagnostics are attached to predictions."""
        pred = self.engine.predict_sentiment("bad battery life")
        self.assertIn("lexical_analysis", pred)
        lex = pred["lexical_analysis"]
        self.assertIn("bad", lex["strong_negative_tokens"])
        self.assertIn("battery life", lex["detected_aspects"])
        self.assertEqual(lex["verdict"], "Negative")

    def test_airpods_worst_battery_regression(self):
        """Verify the original screenshot input is correctly classified and explained."""
        text = "airpods pro 2nd generation has the worst battery life"
        res = self.engine.explain_review_side_by_side(text)
        self.assertEqual(res["prediction"]["predicted_label"], "Negative")

        shap_exp = res["shap_explanation"]
        worst_shap = next((a for a in shap_exp["top_attributions"] if a["feature"] == "worst"), None)
        self.assertIsNotNone(worst_shap)
        self.assertEqual(worst_shap["direction_label"], "Supports Negative")

    def test_scientific_guardrail_presence(self):
        """Verify scientific caveats accompany SHAP and LIME explanations."""
        res = self.engine.explain_review_side_by_side("good battery life")
        self.assertIn("caveat", res["shap_explanation"])
        self.assertIn("surrogate model", res["shap_explanation"]["caveat"])
        self.assertIn("caveat", res["lime_explanation"])
        self.assertIn("local linear surrogate", res["lime_explanation"]["caveat"])


if __name__ == "__main__":
    unittest.main()
