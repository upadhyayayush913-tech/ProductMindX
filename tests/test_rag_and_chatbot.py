"""
Integration Tests for Phase 3, Phase 4 & End-to-End Chatbot Execution.
Tests:
- Vector Store Insertion & Semantic Search
- RAG Context Retrieval
- Conversation Memory
- Intent Classification
- Entity Extraction & Coreference Resolution
- End-to-End Conversation Orchestration
"""

import unittest
from config import CHROMA_DIR
from rag.vector_store import ChromaVectorStore
from rag.retriever import ContextRetriever
from rag.memory import ConversationMemory
from chatbot.intent_classifier import IntentClassifier
from chatbot.entity_extractor import EntityExtractor
from chatbot.conversation_manager import ConversationManager
from data.sample_dataset import SAMPLE_METADATA, SAMPLE_REVIEWS
from nlp_pipeline.pipeline_manager import NLPPipelineManager

class TestRAGAndChatbot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vector_store = ChromaVectorStore()
        cls.pipeline = NLPPipelineManager(vector_store=cls.vector_store)
        cls.insights = cls.pipeline.build_full_knowledge_base()
        cls.products = cls.pipeline.loader.get_all_products()
        cls.conv_manager = ConversationManager(
            products=cls.products,
            vector_store=cls.vector_store,
            nlp_cache=cls.insights
        )

    def test_intent_classification(self):
        ic = IntentClassifier()
        self.assertEqual(ic.classify_intent("What are the pros and cons of Instant Pot?")["intent"], "pros_cons")
        self.assertEqual(ic.classify_intent("Does Sony XM5 have any problems or defects?")["intent"], "issues")
        self.assertEqual(ic.classify_intent("Compare Sony XM5 vs AirPods Pro")["intent"], "comparison")
        self.assertEqual(ic.classify_intent("Hello!")["intent"], "general_chat")

    def test_entity_extraction(self):
        ee = EntityExtractor(products=self.products)
        res = ee.extract_entities("Tell me about the Sony XM5 headphones")
        self.assertIsNotNone(res["primary_product"])
        self.assertIn("WH-1000XM5", res["primary_product"]["title"])

    def test_vector_store_search(self):
        retriever = ContextRetriever(vector_store=self.vector_store)
        context = retriever.retrieve_context("Sony XM5 noise cancellation", top_k=3)
        self.assertTrue(len(context["formatted_context"]) > 0)

    def test_conversation_manager_end_to_end(self):
        # Turn 1: Product inquiry
        turn1 = self.conv_manager.process_message(
            user_query="Tell me about the Instant Pot Duo and its common issues",
            session_id="test_session_1"
        )
        self.assertTrue(len(turn1["reply"]) > 50)
        self.assertEqual(turn1["intent"], "issues")

        # Turn 2: Coreference resolution ("What about its pros?")
        turn2 = self.conv_manager.process_message(
            user_query="What about its pros?",
            session_id="test_session_1",
            conversation_history=[
                {"role": "user", "content": "Tell me about the Instant Pot Duo and its common issues"},
                {"role": "assistant", "content": turn1["reply"]}
            ]
        )
        self.assertIn("Instant Pot", turn2["reply"])

if __name__ == "__main__":
    unittest.main()
