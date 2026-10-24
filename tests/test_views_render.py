"""
Targeted regression test to verify all 5 Streamlit views render cleanly without runtime exceptions:
- render_overview_view
- render_products_view (verifies Best-Supported Products doesn't throw numpy ValueError)
- render_explore_insights_view
- render_ask_productmind_view
- render_system_evaluation_view
"""

import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

from nlp_pipeline.pipeline_manager import NLPPipelineManager
from chatbot.conversation_manager import ConversationManager
from services.product_evidence_service import ProductEvidenceService
import ui.views as views

class TestViewsExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline_manager = NLPPipelineManager()
        cls.conv_manager = ConversationManager()
        cls.conv_manager.evidence_service = ProductEvidenceService(loader=cls.pipeline_manager.loader)
        cls.conv_manager.nlp_cache = cls.pipeline_manager.load_cached_insights()

    def setUp(self):
        # Configure columns mock to automatically return correct number of column mocks
        def fake_columns(spec, *args, **kwargs):
            count = spec if isinstance(spec, int) else len(spec)
            return [MagicMock() for _ in range(count)]
        self.col_patcher = patch("streamlit.columns", side_effect=fake_columns)
        self.col_patcher.start()

    def tearDown(self):
        self.col_patcher.stop()

    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.metric")
    @patch("streamlit.altair_chart")
    @patch("streamlit.divider")
    @patch("streamlit.info")
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "📊 Overview", "messages": []})
    def test_overview_view_render(self, *mocks):
        views.render_overview_view(self.pipeline_manager, self.conv_manager)

    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.tabs", return_value=[MagicMock(), MagicMock()])
    @patch("streamlit.selectbox", return_value="Highest Rating")
    @patch("streamlit.text_input", return_value="")
    @patch("streamlit.button", return_value=False)
    @patch("streamlit.divider")
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "📦 Products", "messages": []})
    def test_products_view_render(self, *mocks):
        views.render_products_view(self.pipeline_manager, self.conv_manager)

    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "🔍 Explore Insights"})
    def test_explore_insights_view_render(self, *mocks):
        with patch("ui.views.render_evidence_intelligence_explorer") as mock_exp:
            views.render_explore_insights_view(self.pipeline_manager, self.conv_manager)
            self.assertTrue(mock_exp.called)

    @patch("streamlit.markdown")
    @patch("streamlit.chat_input", return_value=None)
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "💬 Ask ProductMind", "messages": []})
    def test_ask_productmind_view_render(self, *mocks):
        views.render_ask_productmind_view(self.pipeline_manager, self.conv_manager)

    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.success")
    @patch("streamlit.write")
    @patch("streamlit.button", return_value=False)
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "⚙️ System & Evaluation"})
    def test_system_evaluation_view_render(self, *mocks):
        views.render_system_evaluation_view(self.pipeline_manager, self.conv_manager)

    @patch("streamlit.markdown")
    @patch("streamlit.caption")
    @patch("streamlit.metric")
    @patch("streamlit.button", return_value=False)
    @patch("streamlit.tabs", return_value=[MagicMock(), MagicMock(), MagicMock(), MagicMock()])
    @patch("streamlit.text_area", return_value="airpods pro 2nd generation has the worst battery life")
    @patch("streamlit.session_state", {"category_filter": None, "active_nav": "🔬 NLP Lab"})
    def test_nlp_lab_view_render(self, *mocks):
        views.render_nlp_lab_view(self.pipeline_manager, self.conv_manager)

if __name__ == "__main__":
    unittest.main()
