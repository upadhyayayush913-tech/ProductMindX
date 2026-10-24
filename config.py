"""
Configuration settings for ProductMind X & OmniRoute Integration.

Strict Requirements:
1. Model Pinning: Explicit model identifier (default: groq/llama-3.3-70b-versatile).
2. Auto-routing & silent fallbacks are strictly disabled by default.
3. Credentials kept secure server-side.
4. Grounded RAG with deterministic backend calculations & claim validation.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CHROMA_DIR = CACHE_DIR / "chromadb"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

for directory in [DATA_DIR, CACHE_DIR, CHROMA_DIR, SAMPLE_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load environment variables
load_dotenv(BASE_DIR / ".env")

# Product Categories (Prioritized Amazon Reviews '23 Categories)
CATEGORIES = [
    "Electronics",
    "Cell_Phones_and_Accessories",
    "Home_and_Kitchen",
    "Appliances",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors"
]

# HuggingFace Dataset identifiers
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
SAMPLE_REVIEWS_PER_CATEGORY = int(os.getenv("SAMPLE_REVIEWS_PER_CATEGORY", "1500"))

# NLP & Embedding Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
TRANSFORMER_SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# ChromaDB Collections
CHROMA_COLLECTION_PROFILES = "product_profiles"
CHROMA_COLLECTION_INSIGHTS = "product_insights"
CHROMA_COLLECTION_REVIEWS = "review_chunks"
CHROMA_COLLECTION_MEMORY = "conversation_memory"
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "6"))

# ==============================================================================
# OMNIROUTE GATEWAY & MODEL PINNING POLICY
# ==============================================================================
OMNIROUTE_BASE_URL = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
OMNIROUTE_API_KEY = os.getenv("OMNIROUTE_API_KEY", "sk-omniroute-productmind-key")

# Explicit Pinned Model ID (Provider-qualified as required by OmniRoute)
PINNED_MODEL_ID = os.getenv("PINNED_MODEL_ID", "groq/openai/gpt-oss-120b")

# Strict Governance Policies
ALLOW_MODEL_SWITCHING = False       # Reject unexpected model IDs
FALLBACK_ROUTING_ENABLED = False    # Disable silent fallback routing by default
ALLOW_AUTO_ROUTING = False          # Reject 'auto' models in ProductMind X

# Upstream Provider Keys (Server-side only)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
