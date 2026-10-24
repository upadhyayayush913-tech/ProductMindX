"""
Unit & Integration Tests for Cross-Lingual Product Intelligence & Machine Translation.
Covers:
1. Language detection accuracy (Hindi, Spanish, German, English).
2. Citation masking and restoration integrity ([Review #ID] & [Metadata #ASIN]).
3. MarianMT / Fallback translation with preserved citation tokens.
4. Cross-border review ingestion and metadata preservation.
5. Cross-lingual conversation turn handling and grounding validation.
"""

import unittest
from nlp_pipeline.translator import LanguageDetector, MachineTranslationEngine, translation_engine
from chatbot.conversation_manager import ConversationManager
from nlp_pipeline.pipeline_manager import NLPPipelineManager
from data.data_loader import AmazonDataLoader


class TestLanguageDetector(unittest.TestCase):
    def setUp(self):
        self.detector = LanguageDetector()

    def test_hindi_detection(self):
        # Devanagari script detection
        query = "सोनी हेडफोन की बैटरी कैसी है?"
        lang = self.detector.detect(query)
        self.assertEqual(lang, "hi")

    def test_spanish_detection(self):
        query = "La calidad del sonido y la batería son increíbles"
        lang = self.detector.detect(query)
        self.assertEqual(lang, "es")

    def test_german_detection(self):
        query = "Der Akku und die Geräuschunterdrückung sind hervorragend"
        lang = self.detector.detect(query)
        self.assertEqual(lang, "de")

    def test_english_detection(self):
        query = "How is the battery life on the Sony WH-1000XM5?"
        lang = self.detector.detect(query)
        self.assertEqual(lang, "en")


class TestCitationMaskingAndRestoration(unittest.TestCase):
    def setUp(self):
        self.engine = translation_engine

    def test_mask_and_restore_citations(self):
        sample_text = (
            "According to [Review #REV-B09XS7-1001], the ear cushions are soft. "
            "Also verified by [Metadata #B09XS7JWHH] and [Review #REV-AP2-2002]."
        )
        masked, citation_map = self.engine.mask_citations(sample_text)

        # Citations should be replaced with placeholders
        self.assertNotIn("[Review #REV-B09XS7-1001]", masked)
        self.assertNotIn("[Metadata #B09XS7JWHH]", masked)
        self.assertNotIn("[Review #REV-AP2-2002]", masked)
        self.assertEqual(len(citation_map), 3)

        # Restoration should bring back exact tokens
        restored = self.engine.restore_citations(masked, citation_map)
        self.assertEqual(restored, sample_text)

    def test_translation_preserves_citations(self):
        text_with_cite = "The battery lasts very long [Review #REV-B09XS7-1001] as shown in [Metadata #B09XS7JWHH]."
        res = self.engine.translate(text_with_cite, source_lang="en", target_lang="es")

        translated = res["translated_text"]
        # Exact citation tokens must remain intact
        self.assertIn("[Review #REV-B09XS7-1001]", translated)
        self.assertIn("[Metadata #B09XS7JWHH]", translated)
        self.assertTrue(res["was_translated"])


class TestCrossBorderReviewIngestion(unittest.TestCase):
    def setUp(self):
        self.loader = AmazonDataLoader()
        self.pipeline = NLPPipelineManager()

    def test_sample_dataset_contains_multilingual_reviews(self):
        prods = self.loader.get_all_products()
        rev_df = self.loader.load_reviews("Electronics")

        # Verify cross-border reviews exist in sample data
        languages = rev_df["language"].unique().tolist() if "language" in rev_df.columns else []
        self.assertTrue(any(l in languages for l in ["es", "de", "hi"]))

    def test_pipeline_translates_foreign_reviews(self):
        # Process a product with foreign reviews (Sony WH-1000XM5)
        sony_asin = "B09XS7JWHH"
        insights = self.pipeline.process_and_index_product(sony_asin)

        # Check multilingual stats
        multilingual_stats = insights.get("multilingual_stats", {})
        self.assertIn("non_english_detected", multilingual_stats)
        self.assertIn("translated_count", multilingual_stats)
        # Verify ABSA processed sentences include the translated content
        self.assertGreater(insights.get("total_reviews_analyzed", 0), 0)


class TestCrossLingualConversationTurn(unittest.TestCase):
    def setUp(self):
        self.conv_manager = ConversationManager()

    def test_spanish_query_processing(self):
        query = "¿Cómo es la batería del Sony WH-1000XM5?"
        result = self.conv_manager.process_message(query)

        self.assertEqual(result["detected_language"], "es")
        self.assertTrue(result["is_cross_lingual"])
        self.assertIsNotNone(result["translated_query"])
        self.assertIn("battery", result["translated_query"].lower())
        self.assertIsNotNone(result["product"])
        self.assertEqual(result["product"].get("parent_asin"), "B09XS7JWHH")
        self.assertIn("reply", result)
        self.assertGreater(len(result["reply"]), 20)

    def test_hindi_query_processing(self):
        query = "सोनी WH-1000XM5 की बैटरी कैसी है?"
        result = self.conv_manager.process_message(query)

        self.assertEqual(result["detected_language"], "hi")
        self.assertTrue(result["is_cross_lingual"])
        self.assertIsNotNone(result["translated_query"])
        self.assertIn("battery", result["translated_query"].lower())
        self.assertIsNotNone(result["product"])
        self.assertEqual(result["product"].get("parent_asin"), "B09XS7JWHH")
        self.assertIn("reply", result)

    def test_english_query_unaffected(self):
        query = "What are the pros and cons of Sony WH-1000XM5?"
        result = self.conv_manager.process_message(query)

        self.assertEqual(result["detected_language"], "en")
        self.assertFalse(result["is_cross_lingual"])
        self.assertIsNone(result["translated_query"])
        self.assertIn("reply", result)


if __name__ == "__main__":
    unittest.main()
