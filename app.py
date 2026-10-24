"""
Main Streamlit Application for ProductMind X Enterprise Platform.
AI-First Evidence-Grounded Consumer Intelligence & Emerging Product Defect Detection.
Powered by OmniRoute AI Gateway, Aspect-Based Sentiment Analysis, and Hybrid ChromaDB RAG.
Pixel-Accurate reproduction of the executive analytics dashboard.
"""

import streamlit as st
import uuid

# Page configuration
st.set_page_config(
    page_title="ProductMind X — AI Consumer Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from config import CATEGORIES, PINNED_MODEL_ID
from rag.vector_store import ChromaVectorStore
from nlp_pipeline.pipeline_manager import NLPPipelineManager
from chatbot.conversation_manager import ConversationManager
from services.product_evidence_service import ProductEvidenceService
from services.filter_service import CorpusFilterService, FilterState
from ui.styles import CUSTOM_CSS
from ui.sidebar import render_sidebar
from ui.views import (
    render_overview_view,
    render_products_view,
    render_explore_insights_view,
    render_ask_productmind_view,
    render_nlp_lab_view,
    render_system_evaluation_view,
    render_explore_view,
    render_research_view
)
from utils.logger import get_logger

logger = get_logger("app")

# Apply custom enterprise CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------------------------------------------------
# Initialize Cached Singletons
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing ProductMind X Knowledge Base & OmniRoute Client...")
def initialize_system():
    vector_store = ChromaVectorStore()
    pipeline_manager = NLPPipelineManager(vector_store=vector_store)
    insights = pipeline_manager.build_full_knowledge_base()
    products = pipeline_manager.loader.get_all_products()

    conv_manager = ConversationManager(
        products=products,
        vector_store=vector_store,
        nlp_cache=insights
    )
    # Attach unified evidence service
    conv_manager.evidence_service = ProductEvidenceService(loader=pipeline_manager.loader)
    return vector_store, pipeline_manager, conv_manager

vector_store, pipeline_manager, conv_manager = initialize_system()
filter_service = CorpusFilterService(evidence_service=conv_manager.evidence_service)

# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = f"session_{uuid.uuid4().hex[:6]}"

if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "🔍 Investigate (AI Agent)"

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "applied_filters" not in st.session_state:
    st.session_state["applied_filters"] = FilterState()

# Render Sidebar (Workspace Navigation, Data Filters, System Status, Diagnostics)
active_filter = render_sidebar(conv_manager, filter_service)

# -------------------------------------------------------------
# Main Content Workspace Router (Chatbot-First NLP Research Experience)
# -------------------------------------------------------------
active_nav = st.session_state.get("active_nav", "🔍 Investigate (AI Agent)")

if "Investigate" in active_nav or "Chat" in active_nav:
    render_ask_productmind_view(pipeline_manager, conv_manager)
elif "Explore" in active_nav or "Catalog" in active_nav or "Product" in active_nav or "Issue" in active_nav:
    render_explore_view(pipeline_manager, conv_manager)
elif "Research" in active_nav or "Lab" in active_nav or "Evaluation" in active_nav:
    render_research_view(pipeline_manager, conv_manager)
elif "Overview" in active_nav:
    render_overview_view(pipeline_manager, conv_manager)
else:
    render_ask_productmind_view(pipeline_manager, conv_manager)
