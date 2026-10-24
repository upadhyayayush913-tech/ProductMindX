"""
Pixel-Accurate Enterprise Visual Design System for ProductMind X.
Unified Theme Tokens matching mandatory reference design:
- Main background: #0B1119
- Sidebar background: #0C141E
- Card surface: #111B27
- Elevated surface: #192636
- Borders: #233447
- Primary teal: #35D8B7
- Secondary blue: #60A5FA
- Warning amber: #F5B955
- Critical coral: #F07178
- Purple accent: #A78BFA
- Primary text: #F1F5F9
- Secondary text: #A8B7C9
- Muted text: #718096
"""

CHART_COLORS = {
    "primary_teal": "#35D8B7",
    "secondary_blue": "#60A5FA",
    "warning_amber": "#F5B955",
    "critical_coral": "#F07178",
    "neutral_slate": "#718096",
    "purple_accent": "#A78BFA",
    "grid_color": "rgba(255, 255, 255, 0.05)",
    "text_primary": "#F1F5F9",
    "text_secondary": "#A8B7C9",
    "card_bg": "#111B27",
    "border_color": "#233447"
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-dark: #0B1119;
    --sidebar-bg: #0C141E;
    --card-bg: #111B27;
    --elevated-bg: #192636;
    --border-color: #233447;
    --primary-teal: #35D8B7;
    --primary-hover: #2ec2a4;
    --secondary-blue: #60A5FA;
    --warning-amber: #F5B955;
    --critical-coral: #F07178;
    --purple-accent: #A78BFA;
    --text-primary: #F1F5F9;
    --text-secondary: #A8B7C9;
    --text-muted: #718096;
}

/* Global Application Container */
.stApp {
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

header[data-testid="stHeader"] {
    background-color: rgba(11, 17, 25, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid var(--border-color) !important;
}

/* Main Content Workspace */
.main .block-container {
    max-width: 1360px !important;
    padding-top: 1.25rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2.25rem !important;
    padding-right: 2.25rem !important;
}

/* Fixed-feeling Left Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border-color) !important;
    min-width: 300px !important;
    max-width: 330px !important;
}

/* ============================================================
   SIDEBAR TOP WHITESPACE FIX (Root Cause Elimination)
   Streamlit 1.58 stSidebarHeader creates 84px (3.75rem height + 
   1.5rem margin-bottom). Collapse it completely so branding sits
   cleanly near the top with an intentional ~14px top inset.
   ============================================================ */

/* 1. Eliminate stSidebarHeader and its 84px phantom gap */
[data-testid="stSidebarHeader"],
div[data-testid="stSidebarHeader"],
div[class*="stSidebarHeader"],
header[data-testid="stSidebarHeader"],
.stSidebarHeader {
    display: none !important;
    height: 0px !important;
    min-height: 0px !important;
    max-height: 0px !important;
    margin: 0px !important;
    margin-top: 0px !important;
    margin-bottom: 0px !important;
    padding: 0px !important;
    border: none !important;
    overflow: hidden !important;
    visibility: hidden !important;
}

/* 2. Reset stSidebarContent container top padding */
[data-testid="stSidebarContent"],
div[data-testid="stSidebarContent"],
div[class*="stSidebarContent"],
.stSidebarContent {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

/* 3. Intentional 14px top inset for the user content / branding */
[data-testid="stSidebarUserContent"],
div[data-testid="stSidebarUserContent"],
div[class*="stSidebarUserContent"],
.stSidebarUserContent {
    padding-top: 14px !important;
    padding-left: 18px !important;
    padding-right: 18px !important;
    padding-bottom: 2rem !important;
    margin-top: 0px !important;
}

/* 4. Ensure first vertical block element starts flush */
[data-testid="stSidebarUserContent"] > div:first-child,
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]:first-child,
[data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:first-child {
    padding-top: 0px !important;
    margin-top: 0px !important;
}

/* Tabular Numerals for Consistent Alignment */
[data-testid="stMetricValue"], .tabular-nums, table td, code {
    font-variant-numeric: tabular-nums !important;
}

/* Clean Form Controls */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea > div > div > textarea {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}

.stTextInput > div > div > input:focus,
.stSelectbox > div > div:focus-within,
.stTextArea > div > div > textarea:focus {
    border-color: var(--primary-teal) !important;
    box-shadow: 0 0 0 1px var(--primary-teal) !important;
}

/* Slider Customization */
.stSlider [data-baseweb="slider"] {
    padding-top: 6px !important;
    padding-bottom: 6px !important;
}

.stSlider div[role="slider"] {
    background-color: var(--primary-teal) !important;
    border-color: var(--primary-teal) !important;
}

/* Checkbox Customization */
.stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.86rem !important;
}

/* Primary and Outline Buttons */
.stButton > button {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    transition: all 0.15s ease !important;
}

.stButton > button:hover {
    background-color: var(--elevated-bg) !important;
    border-color: var(--primary-teal) !important;
    color: var(--primary-teal) !important;
}

.stButton > button[kind="primary"] {
    background-color: var(--primary-teal) !important;
    color: #0B1119 !important;
    font-weight: 600 !important;
    border: none !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-hover) !important;
    color: #0B1119 !important;
}

/* Avatar-Free Clean Chat Styling */
.stChatMessage {
    padding: 1.15rem 1.35rem !important;
    border-radius: 8px !important;
    margin-bottom: 0.9rem !important;
    border: 1px solid var(--border-color) !important;
    background-color: var(--card-bg) !important;
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarCustom"],
[data-testid="chatAvatar"],
.stChatMessage [data-testid*="ChatMessageAvatar"],
.stChatMessage [data-testid*="avatar"] {
    display: none !important;
}

/* Expander Header */
.streamlit-expanderHeader {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.streamlit-expanderContent {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-bottom-left-radius: 6px !important;
    border-bottom-right-radius: 6px !important;
    padding: 14px !important;
}

/* ============================================================
   CHATBOT RESPONSE CARD & TYPOGRAPHIC HIERARCHY
   ============================================================ */

.productmind-response-card {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    padding: 18px 22px !important;
    margin-bottom: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-size: 0.98rem !important; /* ~15.7px, in 15-17px body range */
    line-height: 1.62 !important;
}

/* 1. Main Heading Typography (28-32px, Bold, Clean Font, Left-Aligned) */
.productmind-main-title,
.stMarkdown h1 {
    font-size: 1.85rem !important; /* ~29.6px, in 28-32px range */
    font-weight: 750 !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.025em !important;
    line-height: 1.25 !important;
    margin-top: 0.45rem !important;
    margin-bottom: 1.15rem !important;
    text-align: left !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* 2. Section Headings (Observation, Calculation, Defect, Recommendation: 18-22px, Bold) */
.productmind-section-heading,
.stMarkdown h2,
.stMarkdown h3 {
    font-size: 1.25rem !important; /* 20px, in 18-22px range */
    font-weight: 700 !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    letter-spacing: -0.015em !important;
    line-height: 1.35 !important;
    margin-top: 1.3rem !important;
    margin-bottom: 0.55rem !important;
    text-align: left !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.stMarkdown h4 {
    font-size: 1.10rem !important; /* ~17.6px */
    font-weight: 650 !important;
    color: #F1F5F9 !important;
    margin-top: 1.0rem !important;
    margin-bottom: 0.45rem !important;
}

/* 3. Body Text & Paragraphs (15-17px, Readable Line Height) */
.productmind-response-card p,
.stMarkdown p {
    font-size: 0.98rem !important; /* ~15.7px */
    line-height: 1.62 !important;
    color: #E2E8F0 !important;
    margin-bottom: 0.85rem !important;
}

/* 4. Lists (Consistent Indentation & Spacing) */
.productmind-response-card ul,
.productmind-response-card ol,
.stMarkdown ul,
.stMarkdown ol {
    margin-top: 0.4rem !important;
    margin-bottom: 0.85rem !important;
    padding-left: 1.45rem !important;
}

.productmind-response-card li,
.stMarkdown li {
    font-size: 0.98rem !important;
    line-height: 1.60 !important;
    color: #E2E8F0 !important;
    margin-bottom: 0.35rem !important;
}

/* 5. Evidence References & Inline Code Tokens */
.productmind-response-card code,
.stMarkdown code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84rem !important;
    background-color: rgba(53, 216, 183, 0.08) !important;
    color: #35D8B7 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
    border: 1px solid rgba(53, 216, 183, 0.22) !important;
}

/* Blockquote Notices */
.productmind-response-card blockquote,
.stMarkdown blockquote {
    border-left: 3px solid #35D8B7 !important;
    background-color: rgba(53, 216, 183, 0.06) !important;
    padding: 8px 14px !important;
    border-radius: 4px !important;
    margin-top: 0.4rem !important;
    margin-bottom: 0.95rem !important;
    color: #E2E8F0 !important;
    font-size: 0.90rem !important;
}

/* Polished Dark-Themed Markdown Tables */
.stMarkdown table {
    width: 100% !important;
    border-collapse: collapse !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    margin-top: 0.65rem !important;
    margin-bottom: 0.95rem !important;
    background-color: #0E1622 !important;
    font-size: 0.86rem !important;
}

.stMarkdown table th {
    background-color: #162232 !important;
    color: var(--primary-teal) !important;
    font-weight: 600 !important;
    text-align: left !important;
    padding: 9px 12px !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.stMarkdown table td {
    padding: 8px 12px !important;
    border-bottom: 1px solid rgba(35, 52, 71, 0.5) !important;
    color: #E2E8F0 !important;
    line-height: 1.4 !important;
}

.stMarkdown table tr:nth-child(even) td {
    background-color: rgba(25, 38, 54, 0.35) !important;
}

/* Hide Streamlit Menu & Footer for Clean Enterprise Feel */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
