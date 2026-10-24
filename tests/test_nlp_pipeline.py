"""
Unit & Integration Tests for Phase 1 & Phase 2: NLP Pipeline.
Tests:
- Text Preprocessor
- Sentiment Analyzer (VADER & star calibration)
- Aspect Extractor (ABSA)
- Issue & Defect Detector (Severity & Temporal Trends)
- Topic Modeler
"""

import unittest
from data.preprocessor import TextPreprocessor
from nlp_pipeline.sentiment_analyzer import SentimentAnalyzer
from nlp_pipeline.aspect_extractor import AspectExtractor
from nlp_pipeline.issue_detector import IssueDetector
from nlp_pipeline.topic_modeler import ReviewTopicModeler

class TestNLPPipeline(unittest.TestCase):
    def setUp(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment = SentimentAnalyzer(use_transformer=False)
        self.absa = AspectExtractor()
        self.issue_det = IssueDetector()
        self.topic_mod = ReviewTopicModeler(n_topics=2)

    def test_preprocessor(self):
        raw = "I don't think it's good! <p>Visit http://example.com</p>  Awesome!!!!"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("http://example.com", cleaned)
        self.assertIn("do not", cleaned)
        self.assertIn("it is", cleaned)

        sents = self.preprocessor.tokenize_sentences("Battery is great. But the lid leaks steam.")
        self.assertEqual(len(sents), 2)

    def test_sentiment_analyzer(self):
        pos_res = self.sentiment.analyze_text("This product is fantastic and durable!", rating=5.0)
        self.assertEqual(pos_res["final_sentiment"], "positive")
        self.assertGreater(pos_res["vader_compound"], 0.2)

        neg_res = self.sentiment.analyze_text("Terrible quality, stopped working within 2 days.", rating=1.0)
        self.assertEqual(neg_res["final_sentiment"], "negative")
        self.assertLess(neg_res["vader_compound"], -0.2)

    def test_aspect_extraction(self):
        sample_reviews = [
            {"text": "The battery life is phenomenal and sound quality is crisp.", "rating": 5.0},
            {"text": "Battery life lasts forever. Earcups are very comfortable.", "rating": 5.0},
            {"text": "The price is too high and expensive for what you get.", "rating": 2.0}
        ]
        res = self.absa.aggregate_product_aspects(sample_reviews)
        pros_aspects = [p["aspect"].lower() for p in res["pros"]]
        cons_aspects = [c["aspect"].lower() for c in res["cons"]]

        self.assertIn("battery life", pros_aspects)
        self.assertIn("price & value", cons_aspects)

    def test_issue_detection(self):
        neg_reviews = [
            {
                "rating": 1.0,
                "title": "Severe defect",
                "text": "The motor stopped working and makes a loud grinding noise. Smells like smoke.",
                "timestamp": 1690000000000
            },
            {
                "rating": 1.0,
                "title": "Overheating",
                "text": "Spark and burn error appeared after 3 weeks.",
                "timestamp": 1695000000000
            }
        ]
        issues = self.issue_det.detect_product_issues(neg_reviews)
        self.assertTrue(len(issues) > 0)
        # Check that high/critical severity is captured
        severities = [i["severity"] for i in issues]
        self.assertTrue("Critical" in severities or "High" in severities)

    def test_topic_modeling(self):
        texts = [
            "Noise cancellation is amazing on airplanes and trains.",
            "Sound quality has punchy bass and clear vocals.",
            "Battery lasts for 30 hours of continuous music.",
            "Earcups are soft and headband does not hurt."
        ]
        topics = self.topic_mod.fit_transform_reviews(texts)
        self.assertEqual(len(topics), 2)
        self.assertTrue(len(topics[0]["keywords"]) > 0)

if __name__ == "__main__":
    unittest.main()
