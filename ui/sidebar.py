"""
Pixel-Accurate Sidebar for ProductMind X.
Matches mandatory reference design layout:
1. ProductMind X logo / intelligence icon + Subtitle 'AI Product Intelligence'
2. Active 'Overview' navigation item
3. DATA FILTERS section:
   - Category selector
   - Product selector
   - Rating-range slider
   - Date-range selector
   - Sentiment selector
   - Verified purchases checkbox
   - Apply Filters button
   - Active Filters summary + clear control
4. SYSTEM STATUS section:
   - OmniRoute status (Online)
   - Pinned model name (gpt-oss-120b)
5. Collapsible Diagnostics & Settings
"""

import streamlit as st
from typing import Dict, List, Any, Optional

from config import CATEGORIES, PINNED_MODEL_ID
from services.filter_service import FilterState, CorpusFilterService
from utils.logger import get_logger

logger = get_logger("sidebar")

def render_sidebar(conversation_manager, filter_service: CorpusFilterService) -> FilterState:
    """
    Render pixel-accurate left sidebar matching the reference dashboard.
    Returns the currently active FilterState.
    """
    # Initialize session state for applied filters
    if "applied_filters" not in st.session_state:
        st.session_state["applied_filters"] = FilterState()

    applied: FilterState = st.session_state["applied_filters"]

    with st.sidebar:
        # 1. Brand Logo & Subtitle (Prominent, High-Alignment Brand Block)
        st.markdown(
            """
            <div style="display: flex; align-items: flex-start; gap: 11px; margin-top: 0px; margin-bottom: 1.05rem;">
                <div style="display: flex; align-items: center; justify-content: center; padding-top: 2px; flex-shrink: 0;">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#35D8B7" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a7 7 0 0 0-7 7c0 2.5 1.5 4.5 3 6v3h8v-3c1.5-1.5 3-3.5 3-6a7 7 0 0 0-7-7z"></path>
                        <path d="M9 22h6"></path>
                        <path d="M10 12h4"></path>
                    </svg>
                </div>
                <div style="display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 1.88rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.03em; line-height: 1.1; white-space: nowrap;">
                        ProductMind <span style="color: #35D8B7;">X</span>
                    </div>
                    <div style="font-size: 0.86rem; font-weight: 500; color: #94A3B8; margin-top: 3px; letter-spacing: 0.01em;">
                        AI Product Intelligence
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Navigation Selector
        st.markdown(
            """
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; margin-bottom: 0.45rem;">
                NAVIGATION WORKSPACE
            </div>
            """,
            unsafe_allow_html=True
        )
        nav_options = [
            "🔍 Investigate (AI Agent)",
            "🧭 Explore (Catalog & Issues)",
            "🔬 Research (Experiments & XAI)",
            "📊 Overview"
        ]
        legacy_map = {
            "💬 AI Research Chat": "🔍 Investigate (AI Agent)",
            "🔍 Product Intelligence": "🧭 Explore (Catalog & Issues)",
            "📈 Emerging Issues": "🧭 Explore (Catalog & Issues)",
            "🔬 NLP Lab": "🔬 Research (Experiments & XAI)",
            "📑 Evidence & Evaluation": "🔬 Research (Experiments & XAI)",
            "📊 Overview": "📊 Overview"
        }
        current_nav = st.session_state.get("active_nav", "🔍 Investigate (AI Agent)")
        current_nav = legacy_map.get(current_nav, current_nav)
        cur_idx = nav_options.index(current_nav) if current_nav in nav_options else 0
        selected_nav = st.selectbox(
            "Workspace Navigation",
            options=nav_options,
            index=cur_idx,
            label_visibility="collapsed",
            key="sb_nav_workspace"
        )
        st.session_state["active_nav"] = selected_nav

        st.markdown("<div style='height: 0.65rem;'></div>", unsafe_allow_html=True)

        # 2. DATA FILTERS Header
        st.markdown(
            """
            <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; margin-bottom: 0.65rem;">
                DATA FILTERS
            </div>
            """,
            unsafe_allow_html=True
        )

        # Category Filter
        st.markdown("<div style='font-size: 0.78rem; color: #A8B7C9; margin-bottom: 2px;'>Category</div>", unsafe_allow_html=True)
        cat_options = ["All Categories"] + [c.replace("_", " ") for c in CATEGORIES]
        
        # Current index
        curr_cat_disp = "All Categories" if not applied.category else applied.category.replace("_", " ")
        cat_idx = cat_options.index(curr_cat_disp) if curr_cat_disp in cat_options else 0
        
        selected_cat_disp = st.selectbox(
            "Category",
            options=cat_options,
            index=cat_idx,
            label_visibility="collapsed",
            key="sb_category"
        )
        selected_cat = None if selected_cat_disp == "All Categories" else selected_cat_disp.replace(" ", "_")

        # Dynamic Product Filter based on selected category
        st.markdown("<div style='font-size: 0.78rem; color: #A8B7C9; margin-top: 8px; margin-bottom: 2px;'>Product</div>", unsafe_allow_html=True)
        available_products = filter_service.get_products_for_category(selected_cat)
        prod_map = {
            (p.get("parent_asin") or p.get("asin")): p.get("title", "Product")
            for p in available_products
        }
        prod_options = ["All Products"] + list(prod_map.values())
        
        curr_prod_disp = "All Products"
        if applied.product_asin and applied.product_asin in prod_map:
            curr_prod_disp = prod_map[applied.product_asin]
        
        prod_idx = prod_options.index(curr_prod_disp) if curr_prod_disp in prod_options else 0
        selected_prod_disp = st.selectbox(
            "Product",
            options=prod_options,
            index=prod_idx,
            label_visibility="collapsed",
            key="sb_product"
        )
        
        selected_prod_asin = None
        if selected_prod_disp != "All Products":
            for asin, title in prod_map.items():
                if title == selected_prod_disp:
                    selected_prod_asin = asin
                    break

        # Rating Range Slider
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 2px;">
                <span style="font-size: 0.78rem; color: #A8B7C9;">Rating Range</span>
                <span style="font-size: 0.76rem; color: #F1F5F9; font-weight: 600;">{applied.rating_min:.1f} – {applied.rating_max:.1f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        selected_rating = st.slider(
            "Rating Range",
            min_value=1.0,
            max_value=5.0,
            value=(float(applied.rating_min), float(applied.rating_max)),
            step=0.5,
            label_visibility="collapsed",
            key="sb_rating"
        )

        # Sentiment Selector
        st.markdown("<div style='font-size: 0.78rem; color: #A8B7C9; margin-top: 10px; margin-bottom: 2px;'>Sentiment</div>", unsafe_allow_html=True)
        sent_options = ["All Sentiment", "Positive", "Neutral", "Negative"]
        sent_idx = sent_options.index(applied.sentiment) if applied.sentiment in sent_options else 0
        selected_sentiment = st.selectbox(
            "Sentiment",
            options=sent_options,
            index=sent_idx,
            label_visibility="collapsed",
            key="sb_sentiment"
        )

        # Verified purchases checkbox
        verified_only = st.checkbox(
            "Verified purchases only",
            value=applied.verified_only,
            key="sb_verified"
        )

        # Apply Filters Button (with filter funnel icon)
        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
        if st.button("⏚ Apply Filters", key="btn_apply_filters", use_container_width=True):
            st.session_state["applied_filters"] = FilterState(
                category=selected_cat,
                product_asin=selected_prod_asin,
                rating_min=selected_rating[0],
                rating_max=selected_rating[1],
                sentiment=None if selected_sentiment == "All Sentiment" else selected_sentiment,
                verified_only=verified_only
            )
            st.rerun()

        # Active Filters Box with Clear [x] Control
        current_applied = st.session_state["applied_filters"]
        active_pills = current_applied.get_active_descriptions(prod_map)
        
        clear_col1, clear_col2 = st.columns([4, 1])
        with clear_col1:
            st.markdown(
                """
                <div style="font-size: 0.76rem; font-weight: 600; color: #A8B7C9; margin-top: 10px;">
                    Active Filters
                </div>
                """,
                unsafe_allow_html=True
            )
        with clear_col2:
            if not current_applied.is_default():
                if st.button("✕", key="btn_clear_filters", help="Reset all filters to default"):
                    st.session_state["applied_filters"] = FilterState()
                    st.rerun()

        if active_pills:
            pills_html = "".join(
                f"<span style='background-color: rgba(53, 216, 183, 0.12); border: 1px solid rgba(53, 216, 183, 0.3); color: #35D8B7; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; margin-right: 4px; margin-bottom: 4px; display: inline-block;'>{p}</span>"
                for p in active_pills
            )
            st.markdown(
                f"""
                <div style="background-color: #111B27; border: 1px solid #233447; border-radius: 6px; padding: 8px; margin-top: 4px;">
                    {pills_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="background-color: #111B27; border: 1px solid #233447; border-radius: 6px; padding: 8px; margin-top: 4px; font-size: 0.76rem; color: #718096;">
                    None applied
                </div>
                """,
                unsafe_allow_html=True
            )

        # 4. SYSTEM STATUS Header
        st.markdown(
            """
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; margin-top: 1.5rem; margin-bottom: 0.65rem;">
                SYSTEM STATUS
            </div>
            """,
            unsafe_allow_html=True
        )

        health = conversation_manager.response_generator.omniroute_client.check_health()
        is_healthy = health.get("status") == "healthy"
        status_color = "#35D8B7" if is_healthy else "#F5B955"
        status_label = "Online" if is_healthy else "Standby"

        # Model display name (e.g. gpt-oss-120b)
        model_name = PINNED_MODEL_ID.split("/")[-1] if "/" in PINNED_MODEL_ID else PINNED_MODEL_ID

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.82rem;">
                <div style="display: flex; align-items: center; gap: 7px; color: #A8B7C9;">
                    <span style="color: {status_color}; font-size: 0.65rem;">●</span> OmniRoute
                </div>
                <div style="color: {status_color}; font-weight: 600;">
                    {status_label}
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; font-size: 0.82rem;">
                <div style="display: flex; align-items: center; gap: 7px; color: #A8B7C9;">
                    <span style="color: #35D8B7; font-size: 0.65rem;">●</span> Model
                </div>
                <div style="color: #35D8B7; font-weight: 600; font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
                    {model_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # 5. CROSS-LINGUAL TRANSLATION Header
        st.markdown(
            """
            <div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #718096; margin-top: 1.15rem; margin-bottom: 0.55rem;">
                INTELLIGENCE LANGUAGE
            </div>
            """,
            unsafe_allow_html=True
        )

        lang_options = [
            "🌐 Auto-Detect",
            "🇬🇧 English (en)",
            "🇪🇸 Spanish (es)",
            "🇮🇳 Hindi (hi)",
            "🇩🇪 German (de)",
            "🇫🇷 French (fr)"
        ]
        lang_code_map = {
            "🌐 Auto-Detect": "auto",
            "🇬🇧 English (en)": "en",
            "🇪🇸 Spanish (es)": "es",
            "🇮🇳 Hindi (hi)": "hi",
            "🇩🇪 German (de)": "de",
            "🇫🇷 French (fr)": "fr"
        }
        current_target = st.session_state.get("target_language", "auto")
        inv_map = {v: k for k, v in lang_code_map.items()}
        current_label = inv_map.get(current_target, "🌐 Auto-Detect")
        cur_lang_idx = lang_options.index(current_label) if current_label in lang_options else 0

        selected_lang_label = st.selectbox(
            "Language",
            options=lang_options,
            index=cur_lang_idx,
            label_visibility="collapsed",
            key="sb_target_language"
        )
        st.session_state["target_language"] = lang_code_map[selected_lang_label]
        st.caption("HF MarianMT pipeline preserves exact `[Review #ID]` citations.")

        # 6. Collapsible Diagnostics & Settings
        with st.expander("Diagnostics & Settings", expanded=False):
            st.caption("Gateway Authentication:")
            provider = st.selectbox("Provider", ["groq", "google", "openrouter", "openai"], index=0, key="sb_diag_prov")
            user_key = st.text_input(f"{provider.upper()} API Key", type="password", value=st.session_state.get(f"{provider}_key", ""), key="sb_diag_key")
            st.session_state[f"{provider}_key"] = user_key
            if st.button("Update Model Credentials", use_container_width=True, key="btn_update_cred"):
                if user_key:
                    if provider == "groq":
                        new_pinned = "groq/openai/gpt-oss-120b"
                    elif provider == "google":
                        new_pinned = "google/gemini-2.0-flash"
                    elif provider == "openrouter":
                        new_pinned = "openrouter/meta-llama/llama-3.3-70b-instruct:free"
                    else:
                        new_pinned = f"{provider}/gpt-3.5-turbo"
                    conversation_manager.response_generator.update_credentials(pinned_model_id=new_pinned)
                    st.success(f"Model updated: `{new_pinned}`")
                else:
                    st.warning("Please enter a key.")

            st.divider()
            if st.button("Clear Conversation History", use_container_width=True, key="btn_clear_hist"):
                conversation_manager.reset_conversation()
                st.session_state["messages"] = []
                st.success("History cleared.")
                st.rerun()

    return st.session_state["applied_filters"]
