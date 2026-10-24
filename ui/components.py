"""
Reusable UI Components for ProductMind X Enterprise Platform.
Implements:
1. Reusable Product DNA Component (Official vs Review-inferred, ABSA, issues, limitations)
2. Evidence Intelligence Explorer (Aspects, issues, timelines, supporting/contradictory evidence)
3. Grounded Claim Validation & Deterministic Analytics Card
4. Technical Gateway Diagnostics & ChromaDB Trace
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from config import PINNED_MODEL_ID

# ---------------------------------------------------------------------------
# 1. Reusable Product DNA Component
# ---------------------------------------------------------------------------

def render_product_dna(
    product: Dict[str, Any],
    nlp_insights: Optional[Dict[str, Any]] = None,
    reviews: Optional[List[Dict[str, Any]]] = None
):
    """
    Renders the unified Product DNA card:
    - Official manufacturer metadata & features
    - Review-inferred aspects (kept strictly separate from manufacturer features)
    - Aspect sentiment breakdown with sample sizes
    - Evidence-qualified issue signals
    - Representative positive & negative reviews
    - Data coverage & historical price disclaimers
    """
    if not product:
        st.warning("No product selected for Product DNA inspection.")
        return

    nlp_insights = nlp_insights or {}
    reviews = reviews or []

    title = product.get("title", "Product")
    asin = product.get("parent_asin") or product.get("asin", "N/A")
    brand = product.get("store", "N/A")
    price = product.get("price", "Not available")
    category = str(product.get("main_category", "")).replace("_", " ")
    source_rating = product.get("average_rating", 0.0)
    source_rating_count = product.get("rating_number", 0)
    features = product.get("features", [])
    desc = product.get("description", "")

    st.markdown(f"### 🧬 Product DNA: {title}")
    st.caption(f"ASIN: `{asin}` • Brand: **{brand}** • Category: **{category}**")

    # Top KPI Metrics Strip
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Catalog Rating", f"⭐ {source_rating} / 5.0", f"{source_rating_count:,} source reviews")
    col2.metric("Indexed Sample", f"{len(reviews)} reviews", f"{sum(1 for r in reviews if r.get('verified_purchase', True))} verified")
    col3.metric("List Price", price, "Historical catalog price*")
    
    pros = nlp_insights.get("pros", [])
    cons = nlp_insights.get("cons", [])
    issues = nlp_insights.get("issues", [])
    col4.metric("Issue Signals", f"{len(issues)} detected", f"{len(pros)} strengths / {len(cons)} weaknesses")

    st.caption("*Historical source price from Amazon Reviews '23 corpus; not guaranteed current market price.")

    # Two-Column DNA Matrix: Official Features vs. Review-Inferred Aspects
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("#### 📜 Official Manufacturer Features (Amazon Metadata)")
        from services.product_evidence_service import is_metadata_field_present
        if is_metadata_field_present(features):
            if hasattr(features, "__iter__") and not isinstance(features, (str, bytes, dict)):
                for f in features:
                    st.markdown(f"- {f}")
            else:
                st.markdown(str(features))
        else:
            st.info("Manufacturer feature bullet points not available in metadata record.")

        if is_metadata_field_present(desc):
            with st.expander("Product Description & Specs", expanded=False):
                st.write(str(desc))
                details = product.get("details")
                if is_metadata_field_present(details):
                    st.code(str(details), language="json")

    with c_right:
        st.markdown("#### 🧠 Review-Inferred Aspects (ABSA Analytics)")
        st.caption("Extracted from customer reviews (distinct from manufacturer claims):")

        if pros:
            st.markdown("**Perceived Strengths:**")
            for p in pros[:3]:
                st.markdown(
                    f"- **{p['aspect']}**: `{p['positive_percentage']}% positive` "
                    f"({p['mention_count']} mentions, score: +{p['sentiment_score']})"
                )
        if cons:
            st.markdown("**Perceived Weaknesses:**")
            for c in cons[:3]:
                st.markdown(
                    f"- **{c['aspect']}**: `{100 - c['positive_percentage']:.0f}% negative` "
                    f"({c['mention_count']} mentions, score: {c['sentiment_score']})"
                )

        if not pros and not cons:
            st.info("Aspect sentiment models have not yet processed review chunks for this product.")

    # Issue Signals Section
    if issues:
        st.markdown("#### ⚠️ Evidence-Qualified Issue Signals")
        for iss in issues:
            sev = iss.get("severity", "Medium")
            trend = iss.get("trend", "Stable")
            badge = "badge-critical" if sev == "Critical" else ("badge-high" if sev == "High" else "badge-medium")
            trend_badge = '<span class="badge-emerging">⚠️ Increasing</span>' if "Increasing" in trend else f'<span class="badge-stable">{trend}</span>'

            st.markdown(f"""
            <div class="insight-card-issue">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong>{iss['issue']}</strong>
                    <div>
                        <span class="{badge}">{sev.upper()}</span>
                        {trend_badge}
                    </div>
                </div>
                <div>Frequency: <strong>{iss['frequency']} reported complaints</strong> | Timeline: {iss.get('first_reported')} to {iss.get('last_reported')}</div>
            </div>
            """, unsafe_allow_html=True)

    # Representative Evidence Passages
    if reviews:
        with st.expander(f"📑 Representative Customer Reviews ({len(reviews)} records)", expanded=False):
            pos_revs = [r for r in reviews if r.get("rating", 3.0) >= 4.0]
            neg_revs = [r for r in reviews if r.get("rating", 3.0) <= 2.0]

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.markdown("##### Supporting Positive Evidence")
                for r in pos_revs[:2]:
                    vp = "✅ Verified" if r.get("verified_purchase", True) else "⚪ Unverified"
                    trans_badge = ""
                    if r.get("translated") or (r.get("language") and r.get("language") != "en"):
                        l_code = r.get("language", "foreign").upper()
                        trans_badge = f" <span style='background-color: rgba(53, 216, 183, 0.12); border: 1px solid rgba(53, 216, 183, 0.3); color: #35D8B7; padding: 1px 6px; border-radius: 4px; font-size: 0.70rem;'>🌐 Translated from {l_code}</span>"
                    st.markdown(f"**`[{r.get('review_id', 'REV')}]`** ⭐ {r.get('rating')}/5.0 ({vp}){trans_badge}", unsafe_allow_html=True)
                    st.caption(f"\"{r.get('text', '')}\"")
                    if r.get("original_text") and r.get("original_text") != r.get("text"):
                        st.caption(f"*Original ({r.get('language', 'foreign').upper()}):* \"{r.get('original_text')}\"")
                    st.divider()

            with r_col2:
                st.markdown("##### Customer Criticisms & Flaws")
                for r in neg_revs[:2]:
                    vp = "✅ Verified" if r.get("verified_purchase", True) else "⚪ Unverified"
                    trans_badge = ""
                    if r.get("translated") or (r.get("language") and r.get("language") != "en"):
                        l_code = r.get("language", "foreign").upper()
                        trans_badge = f" <span style='background-color: rgba(53, 216, 183, 0.12); border: 1px solid rgba(53, 216, 183, 0.3); color: #35D8B7; padding: 1px 6px; border-radius: 4px; font-size: 0.70rem;'>🌐 Translated from {l_code}</span>"
                    st.markdown(f"**`[{r.get('review_id', 'REV')}]`** ⭐ {r.get('rating')}/5.0 ({vp}){trans_badge}", unsafe_allow_html=True)
                    st.caption(f"\"{r.get('text', '')}\"")
                    if r.get("original_text") and r.get("original_text") != r.get("text"):
                        st.caption(f"*Original ({r.get('language', 'foreign').upper()}):* \"{r.get('original_text')}\"")
                    st.divider()


# ---------------------------------------------------------------------------
# 2. Evidence Intelligence Explorer
# ---------------------------------------------------------------------------

def render_evidence_intelligence_explorer(
    all_products: Any = None,
    nlp_cache: Any = None,
    loader: Any = None
):
    """
    Consolidated investigation workspace replacing the legacy knowledge graph:
    - What issue or aspect is involved?
    - Which products are affected?
    - How frequently does the issue occur?
    - Is its frequency or negative-sentiment rate increasing against a valid historical baseline?
    - Which reviews support the finding?
    - What evidence contradicts it?
    - What are the sample size and observation period?
    - What limitations affect the conclusion?
    - What should the user investigate next?
    """
    # Safe auto-resolution if pipeline_manager or conv_manager is passed
    if hasattr(all_products, "loader"):
        pipe = all_products
        conv = nlp_cache
        loader = getattr(pipe, "loader", None) or loader
        nlp_cache = getattr(pipe, "insights_cache", {}) or (getattr(conv, "nlp_cache", {}) if conv else {})
        all_products = loader.get_all_products() if loader else []
    elif all_products is None:
        from data.data_loader import AmazonDataLoader
        loader = AmazonDataLoader()
        all_products = loader.get_all_products()
        nlp_cache = {}
    st.markdown("### 🔍 Evidence Intelligence Explorer")
    st.caption("Practical defect investigation, aspect sentiment trajectories, and supporting/contradictory evidence.")

    # Aggregate all issues across all products
    all_issues = []
    for p in all_products:
        asin = p.get("parent_asin") or p.get("asin")
        ins = nlp_cache.get(asin, {})
        for iss in ins.get("issues", []):
            item = dict(iss)
            item["product_title"] = p.get("title")
            item["parent_asin"] = asin
            item["category"] = p.get("main_category")
            all_issues.append(item)

    # Filter Toolbar
    f_col1, f_col2, f_col3 = st.columns(3)
    categories = ["All Categories"] + sorted(list({p.get("main_category", "") for p in all_products if p.get("main_category")}))
    chosen_cat = f_col1.selectbox("Filter Category", categories)
    
    severities = ["All Severities", "Critical", "High", "Medium"]
    chosen_sev = f_col2.selectbox("Severity Level", severities)
    
    trajectories = ["All Trajectories", "Increasing (Emerging Issue ⚠️)", "Stable", "Recent Detection"]
    chosen_traj = f_col3.selectbox("Trajectory / Trend", trajectories)

    # Apply filters
    filtered_issues = all_issues
    if chosen_cat != "All Categories":
        filtered_issues = [i for i in filtered_issues if i.get("category") == chosen_cat]
    if chosen_sev != "All Severities":
        filtered_issues = [i for i in filtered_issues if i.get("severity") == chosen_sev]
    if chosen_traj != "All Trajectories":
        filtered_issues = [i for i in filtered_issues if chosen_traj in i.get("trend", "")]

    st.markdown(f"**Found {len(filtered_issues)} qualified issue patterns:**")

    if not filtered_issues:
        st.markdown(
            f"""
            <div style="background-color: #1B2430; border: 1px solid rgba(69, 214, 181, 0.25); border-radius: 8px; padding: 24px; text-align: center; margin-top: 14px;">
                <div style="font-size: 1.6rem; margin-bottom: 6px;">🛡️</div>
                <div style="color: #45D6B5; font-weight: 600; font-size: 1.02rem;">
                    No Defect Patterns Matched: {chosen_cat.replace('_', ' ')}
                </div>
                <div style="color: #A8B3C2; font-size: 0.84rem; margin-top: 6px; max-width: 620px; margin-left: auto; margin-right: auto;">
                    Customer reviews in this selection indicate clean operational performance with no critical defect clusters matching the selected filters. Total tracked defect patterns across catalog: {len(all_issues)}.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # Render each qualified issue with investigative breakdown
    for idx, iss in enumerate(filtered_issues):
        sev = iss.get("severity", "Medium")
        trend = iss.get("trend", "Stable")
        badge = "badge-critical" if sev == "Critical" else ("badge-high" if sev == "High" else "badge-medium")
        trend_badge = '<span class="badge-emerging">⚠️ Increasing (Emerging Issue)</span>' if "Increasing" in trend else f'<span class="badge-stable">{trend}</span>'

        with st.expander(f"⚠️ {iss['issue']} — {iss['product_title']} ({sev.upper()} Severity)", expanded=(idx == 0)):
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Product:** `{iss['parent_asin']}`\n\n{iss['product_title'][:40]}...")
            c2.markdown(f"**Severity:** <span class='{badge}'>{sev.upper()}</span>", unsafe_allow_html=True)
            c3.markdown(f"**Trajectory:** {trend_badge}", unsafe_allow_html=True)
            c4.markdown(f"**Frequency:** `{iss['frequency']} complaints`\n\nTimeline: {iss.get('first_reported')} to {iss.get('last_reported')}")

            # Load actual product reviews for supporting and contradictory evidence
            asin = iss["parent_asin"]
            cat = iss.get("category", "Electronics")
            try:
                rev_df = loader.load_reviews(cat)
                matched_rows = rev_df[rev_df["parent_asin"] == asin].to_dict("records")
            except Exception:
                matched_rows = []

            # Supporting complaints
            sample_quotes = iss.get("sample_quotes", [])
            st.markdown("##### 📌 Supporting Review Evidence (Reported Complaints)")
            if sample_quotes:
                for q in sample_quotes:
                    st.markdown(f"- 💬 *\"{q}\"*")
            else:
                st.caption("No direct quotes indexed.")

            # Contradictory evidence (positive reviews on the same product)
            st.markdown("##### ⚖️ Contradictory Evidence & Nuance")
            positive_counter = [r for r in matched_rows if r.get("rating", 3.0) >= 4.0]
            if positive_counter:
                counter = positive_counter[0]
                vp = "Verified Purchase" if counter.get("verified_purchase", True) else "Unverified"
                st.markdown(f"- ⭐ **{counter.get('rating')}/5.0 ({vp}):** *\"{counter.get('text', '')[:200]}...\"*")
            else:
                st.caption("No contradictory positive reviews found in current sample.")

            # Analytical Limitations & Next Investigation Action
            st.markdown("##### 🔍 Investigation Guidance & Limitations")
            st.info(
                f"**Data Scope:** Analysis reflects {len(matched_rows)} indexed review records between {iss.get('first_reported')} and {iss.get('last_reported')}.\n\n"
                f"**Recommended Next Step:** Query Ask ProductMind: *\"What are all the pros and cons of {iss['product_title']}?\"* or *\"Compare {iss['product_title']} with other options.\"*"
            )


# ---------------------------------------------------------------------------
# 3. Grounded Claim Validation Card
# ---------------------------------------------------------------------------

def render_grounded_validation_card(claims_validation: Dict[str, Any], backend_metrics: Dict[str, Any]):
    """Render evidence grounding metrics, citation verification, and backend calculations."""
    if not claims_validation and not backend_metrics:
        return

    score = claims_validation.get("grounding_score_pct", 100.0)
    total_cites = claims_validation.get("total_citations", 0)
    valid_cites = claims_validation.get("valid_citations_count", 0)
    snippets = claims_validation.get("verified_evidence_snippets", [])

    st.markdown("#### 🛡️ Grounded AI Evidence & Claim Validation")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Grounding Score", f"{score}%", "Verified" if score >= 80 else "Caution")
    col2.metric("Valid Citations", f"{valid_cites} / {total_cites}" if total_cites > 0 else "Verified Specs")
    col3.metric("Sample Size", f"{backend_metrics.get('sample_size', backend_metrics.get('total_reviews', 0))} reviews", f"{backend_metrics.get('verified_ratio_pct', 0)}% verified")
    col4.metric("Computed Mean", f"⭐ {backend_metrics.get('average_rating', 0.0)} / 5.0")

    # Star distribution progress bars
    dist = backend_metrics.get("star_distribution_pct", backend_metrics.get("rating_distribution_pct", {}))
    if dist:
        with st.expander("📊 Deterministic Rating Breakdown (Calculated)", expanded=False):
            for star in [5, 4, 3, 2, 1]:
                pct = dist.get(star, 0.0)
                st.write(f"{star}★ ({pct}%):")
                st.progress(pct / 100.0)

    # Verified review snippets
    if snippets:
        with st.expander(f"📑 Verified Source Review Evidence ({len(snippets)} records cited)", expanded=False):
            for s in snippets:
                vp = "✅ Verified Purchase" if s.get("verified_purchase") else "⚪ Unverified"
                trans_badge = ""
                if s.get("translated") or (s.get("language") and s.get("language") != "en"):
                    l_code = s.get("language", "foreign").upper()
                    trans_badge = f" | <span style='background-color: rgba(53, 216, 183, 0.12); border: 1px solid rgba(53, 216, 183, 0.3); color: #35D8B7; padding: 1px 6px; border-radius: 4px; font-size: 0.70rem;'>🌐 Translated from {l_code}</span>"
                st.markdown(f"**`[{s['citation_id']}]`** ⭐ {s.get('rating')}/5.0 | *{vp}*{trans_badge}", unsafe_allow_html=True)
                st.caption(f"\"{s.get('text')}\"")
                if s.get("original_text") and s.get("original_text") != s.get("text"):
                    st.caption(f"*Original ({s.get('language', 'foreign').upper()}):* \"{s.get('original_text')}\"")
                st.divider()

    # Hallucinated citations alert
    hallucinated = claims_validation.get("hallucinated_review_citations", [])
    if hallucinated:
        st.error(f"⚠️ Unverified Citations Detected: {', '.join(hallucinated)}. Flagged as ungrounded.")


# ---------------------------------------------------------------------------
# 4. Diagnostics Footer
# ---------------------------------------------------------------------------

def render_diagnostics_footer(metadata: Dict[str, Any]):
    """Render technical details about RAG retrieval, intent confidence, and OmniRoute gateway."""
    with st.expander("🛠️ Technical Gateway Diagnostics & ChromaDB Trace", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Intent Detected:** `{metadata.get('intent', 'N/A')}`")
        c2.markdown(f"**Pinned Model:** `{metadata.get('pinned_model', PINNED_MODEL_ID)}`")
        c3.markdown(f"**Gateway:** `{metadata.get('engine', 'OmniRoute')}`")

        if metadata.get("is_cross_lingual"):
            st.markdown(
                f"**Cross-Lingual Engine:** Detected `{metadata.get('detected_language')}` ➔ Normalized to English for RAG ➔ Output `{metadata.get('effective_language')}`"
            )

        retrieval_info = metadata.get("retrieval_info", {})
        if retrieval_info:
            st.markdown(
                f"**Retrieval Hits:** Profiles: {retrieval_info.get('profiles_retrieved', 0)} | "
                f"Insights: {retrieval_info.get('insights_retrieved', 0)} | "
                f"Reviews: {retrieval_info.get('reviews_retrieved', 0)}"
            )
            raw_context = retrieval_info.get("formatted_context", "")
            if raw_context:
                with st.expander("View Raw Injected Context", expanded=False):
                    st.code(raw_context, language="text")

# Backwards compatibility wrappers
def render_product_card(product: Dict[str, Any]):
    """Compatibility wrapper redirecting to render_product_dna."""
    return render_product_dna(product)

def render_nlp_insights_tabs(insights: Dict[str, Any]):
    """Compatibility wrapper for legacy tab view."""
    return None

__all__ = [
    "render_product_dna",
    "render_product_card",
    "render_nlp_insights_tabs",
    "render_evidence_intelligence_explorer",
    "render_grounded_validation_card",
    "render_diagnostics_footer"
]
