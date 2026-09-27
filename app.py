"""
Merchant Risk Intelligence
--------------------------
Professional fintech-style merchant risk assessment dashboard.

IMPORTANT:
- XGBoost model and prediction logic remain unchanged.
- OpenAI is used only as an AI Risk Analyst layer.
- The AI does NOT determine or modify the ML risk classification.
"""

import re

import numpy as np
import streamlit as st
import pandas as pd
import joblib

from src.ai_risk_analyst import analyze_merchant_risk


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "models/merchant_risk_xgboost.pkl"

MODEL_NAME = "XGBoost"
TEST_ACCURACY = 0.896
HIGH_RISK_PRECISION = 0.647
HIGH_RISK_RECALL = 0.786
HIGH_RISK_TP = 11
HIGH_RISK_FP = 6
HIGH_RISK_FN = 3

BUSINESS_CATEGORIES = [
    "Apparel",
    "Crypto/Forex-adjacent",
    "Digital Goods/Gaming",
    "Education",
    "Electronics",
    "Groceries",
    "Healthcare",
    "Restaurants",
    "Travel",
    "Utilities",
]

BUSINESS_TYPES = [
    "Private Limited",
    "LLP",
    "Partnership",
    "Sole Proprietorship",
]

KYC_LEVELS = ["Basic", "Enhanced", "Full"]

REGISTERED_STATES = [
    "Maharashtra",
    "Rajasthan",
    "Gujarat",
    "Karnataka",
    "Tamil Nadu",
    "Delhi",
    "Uttar Pradesh",
    "West Bengal",
    "Telangana",
    "Kerala",
]

# Columns required in a batch-upload CSV — matches the keys collected
# from the single-merchant form, in the same order the pipeline
# was trained on.
REQUIRED_COLUMNS = [
    "business_category",
    "business_type",
    "business_age_months",
    "kyc_verification_level",
    "registered_state",
    "website_verified",
    "avg_monthly_txn_volume",
    "avg_transaction_value",
    "txn_growth_rate_30d",
    "txn_success_rate",
    "refund_ratio",
    "chargeback_ratio",
    "night_time_txn_percentage",
    "settlement_delay_requests",
    "bank_account_age_days",
    "reserve_balance_ratio",
    "multiple_bank_accounts_flag",
    "previous_suspension_flag",
    "complaint_count_90d",
    "blacklist_match_score",
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Merchant Risk Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM STYLING
# ============================================================

def inject_custom_css() -> None:

    st.markdown(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

        /* ----------------------------------------------------
           TOKENS
        ---------------------------------------------------- */

        :root {
            --bg: #0a0d13;
            --panel: #10141c;
            --panel-raised: #141924;
            --line: #232a38;
            --line-soft: #1a1f2b;
            --text: #e7ecf3;
            --text-dim: #b7bfd1;
            --text-faint: #7c8598;
            --accent: #4c8dff;
            --low: #2fd08a;
            --medium: #f2a93b;
            --high: #ff5468;
        }

        /* ----------------------------------------------------
           GLOBAL
        ---------------------------------------------------- */

        .stApp {
            background:
                radial-gradient(ellipse 900px 500px at 15% -10%, rgba(76,141,255,0.07), transparent),
                var(--bg);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", -apple-system, "Segoe UI", sans-serif;
            color: var(--text);
        }

        h1, h2, h3, h4 { color: var(--text) !important; }

        h1 {
            font-size: 1.9rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }

        h2 {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }

        h3 {
            font-size: 0.98rem !important;
            font-weight: 600 !important;
            font-family: "IBM Plex Mono", monospace !important;
        }

        p { line-height: 1.6; color: var(--text-dim); }

        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] ul,
        [data-testid="stMarkdownContainer"] ol {
            color: var(--text-dim);
            line-height: 1.6;
        }

        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stMarkdownContainer"] b {
            color: var(--text);
            font-weight: 600;
        }

        code, .stCode, [data-testid="stMetricValue"] {
            font-family: "IBM Plex Mono", monospace !important;
        }

        /* Streamlit native inputs on dark surface */
        [data-testid="stSelectbox"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSlider"] label,
        [data-testid="stRadio"] label,
        [data-testid="stCheckbox"] label,
        .stSelectbox label, .stNumberInput label, .stSlider label {
            color: var(--text-dim) !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }

        [data-baseweb="select"] > div,
        input, textarea {
            background-color: var(--panel-raised) !important;
            border-color: var(--line) !important;
            color: var(--text) !important;
        }

        [data-testid="stWidgetLabel"] p { color: var(--text-dim) !important; }


        /* ----------------------------------------------------
           HEADER
        ---------------------------------------------------- */

        .app-eyebrow {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.72rem;
            color: var(--accent);
            letter-spacing: 0.02em;
            margin-bottom: 0.35rem;
        }

        .app-title {
            font-size: 1.9rem;
            font-weight: 600;
            color: var(--text);
            letter-spacing: -0.015em;
        }

        .app-subtitle {
            color: var(--text-dim);
            font-size: 0.95rem;
            margin-top: 0.3rem;
        }

        .system-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 0.4rem 0.7rem;
            color: var(--text-dim);
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.72rem;
        }

        .status-dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--low);
            box-shadow: 0 0 6px var(--low);
            display: inline-block;
        }


        /* ----------------------------------------------------
           SECTION LABELS
        ---------------------------------------------------- */

        .section-label {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            color: var(--text);
            font-size: 0.98rem;
            font-weight: 600;
            padding-bottom: 0.6rem;
            margin-bottom: 1.1rem;
            border-bottom: 1px solid var(--line);
        }

        .section-label::before {
            content: "";
            width: 3px;
            height: 15px;
            background: var(--accent);
            display: inline-block;
        }

        .section-description {
            color: var(--text-faint);
            font-size: 0.85rem;
            margin-top: -0.6rem;
            margin-bottom: 1.2rem;
        }


        /* ----------------------------------------------------
           RISK RESULT
        ---------------------------------------------------- */

        .risk-container {
            border: 1px solid var(--line);
            border-radius: 6px;
            background: var(--panel);
            padding: 1.5rem 1.7rem;
            margin-top: 0.8rem;
            position: relative;
            overflow: hidden;
        }

        .risk-container::before {
            content: "";
            position: absolute;
            left: 0; top: 0; bottom: 0;
            width: 4px;
        }

        .risk-label {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.72rem;
            color: var(--text-faint);
            letter-spacing: 0.04em;
            margin-bottom: 0.3rem;
        }

        .risk-value {
            font-family: "IBM Plex Mono", monospace;
            font-size: 2.1rem;
            font-weight: 600;
            letter-spacing: -0.01em;
        }

        .risk-description {
            color: var(--text-dim);
            font-size: 0.88rem;
            margin-top: 0.35rem;
            max-width: 60ch;
        }

        .risk-high::before { background: var(--high); }
        .risk-high .risk-value { color: var(--high); }

        .risk-medium::before { background: var(--medium); }
        .risk-medium .risk-value { color: var(--medium); }

        .risk-low::before { background: var(--low); }
        .risk-low .risk-value { color: var(--low); }


        /* ----------------------------------------------------
           PROBABILITY CARDS / BARS
        ---------------------------------------------------- */

        .prob-card {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 6px;
            padding: 0.95rem 1.05rem;
        }

        .prob-title {
            color: var(--text-faint);
            font-size: 0.72rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .prob-value {
            font-family: "IBM Plex Mono", monospace;
            color: var(--text);
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }

        .prob-bars {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 6px;
            padding: 1.1rem 1.3rem 1rem 1.3rem;
        }

        .prob-bar-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.7rem;
        }

        .prob-bar-row:last-child { margin-bottom: 0; }

        .prob-bar-name {
            width: 70px;
            font-size: 0.8rem;
            color: var(--text-dim);
            flex-shrink: 0;
        }

        .prob-bar-track {
            flex: 1;
            height: 8px;
            background: var(--line-soft);
            border-radius: 2px;
            overflow: hidden;
        }

        .prob-bar-fill { height: 100%; border-radius: 2px; }

        .prob-bar-pct {
            font-family: "IBM Plex Mono", monospace;
            width: 58px;
            text-align: right;
            font-size: 0.82rem;
            color: var(--text);
            flex-shrink: 0;
        }

        .shap-bar-row {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            margin-bottom: 0.65rem;
        }

        .shap-bar-row:last-child { margin-bottom: 0; }

        .shap-bar-name {
            width: 200px;
            font-size: 0.8rem;
            color: var(--text-dim);
            flex-shrink: 0;
        }

        .shap-bar-track {
            flex: 1;
            height: 8px;
            background: var(--line-soft);
            border-radius: 2px;
            overflow: hidden;
        }

        .shap-bar-fill { height: 100%; border-radius: 2px; }

        .shap-bar-val {
            font-family: "IBM Plex Mono", monospace;
            width: 70px;
            text-align: right;
            font-size: 0.78rem;
            flex-shrink: 0;
        }


        /* ----------------------------------------------------
           THRESHOLD CARD
        ---------------------------------------------------- */

        .threshold-card {
            background: var(--panel);
            border: 1px dashed var(--line);
            border-radius: 6px;
            padding: 0.9rem 1.1rem;
            margin-top: 0.8rem;
        }

        .threshold-title {
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.78rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.2rem;
        }

        .threshold-text {
            font-size: 0.85rem;
            color: var(--text-faint);
        }


        /* ----------------------------------------------------
           AI ANALYST
        ---------------------------------------------------- */

        .ai-container {
            background: var(--panel);
            border: 1px solid var(--line);
            border-left: 3px solid var(--accent);
            border-radius: 6px;
            padding: 1.3rem 1.5rem;
            margin-top: 1rem;
        }

        .ai-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--text);
        }

        .ai-subtitle {
            color: var(--text-faint);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }

        .ai-status {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid var(--line);
            color: var(--low);
            padding: 0.3rem 0.6rem;
            border-radius: 4px;
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.72rem;
            font-weight: 500;
        }


        /* ----------------------------------------------------
           SIGNAL CARDS
        ---------------------------------------------------- */

        .signal-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 4px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.55rem;
            color: var(--text-dim);
            font-size: 0.87rem;
        }

        .signal-marker { color: var(--high); font-weight: 700; }


        /* ----------------------------------------------------
           HUMAN REVIEW
        ---------------------------------------------------- */

        .review-card {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 0.95rem 1.1rem;
            margin-top: 0.8rem;
        }

        .review-title { font-weight: 600; color: var(--text); }

        .review-text {
            color: var(--text-faint);
            font-size: 0.85rem;
            margin-top: 0.3rem;
        }


        /* ----------------------------------------------------
           SIDEBAR
        ---------------------------------------------------- */

        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        .sidebar-brand {
            font-size: 1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.15rem;
        }

        .sidebar-caption {
            color: var(--text-faint);
            font-size: 0.8rem;
            line-height: 1.5;
        }

        .sidebar-item {
            padding: 0.6rem 0;
            border-bottom: 1px solid var(--line-soft);
        }

        .sidebar-label {
            color: var(--text-faint);
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .sidebar-value {
            font-family: "IBM Plex Mono", monospace;
            color: var(--text);
            font-weight: 500;
            font-size: 0.86rem;
            margin-top: 0.15rem;
        }


        /* ----------------------------------------------------
           BUTTON
        ---------------------------------------------------- */

        .stButton > button {
            border-radius: 4px;
            font-weight: 700;
            min-height: 2.8rem;
            background: var(--accent);
            border: 1px solid var(--accent);
            color: #ffffff;
            letter-spacing: 0.01em;
        }

        .stButton > button p,
        .stButton > button div,
        .stButton > button span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        .stButton > button:hover {
            background: #6ba0ff;
            border-color: #6ba0ff;
        }


        /* ----------------------------------------------------
           METRICS
        ---------------------------------------------------- */

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            padding: 0.8rem 1rem;
            border-radius: 6px;
        }

        [data-testid="stMetricLabel"] { color: var(--text-faint) !important; }


        /* ----------------------------------------------------
           EXPANDER
        ---------------------------------------------------- */

        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 6px !important;
            background: var(--panel) !important;
        }


        /* ----------------------------------------------------
           TABS
        ---------------------------------------------------- */

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stTabs"] [data-baseweb="tab"] {
            color: var(--text-faint);
            font-family: "IBM Plex Mono", monospace;
            font-size: 0.82rem;
            font-weight: 500;
            padding: 0.6rem 0.2rem;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--text) !important;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }


        /* ----------------------------------------------------
           FILE UPLOADER / DOWNLOAD BUTTONS
        ---------------------------------------------------- */

        [data-testid="stFileUploader"] section {
            background: var(--panel-raised);
            border: 1px dashed var(--line);
            border-radius: 6px;
        }

        [data-testid="stFileUploaderDropzone"] { color: var(--text-dim); }

        [data-testid="stDownloadButton"] > button {
            border-radius: 4px;
            border: 1px solid var(--accent);
            background: transparent;
            color: var(--accent);
            font-weight: 600;
        }

        [data-testid="stDownloadButton"] > button:hover {
            background: rgba(76,141,255,0.1);
        }

        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 6px; }


        /* ----------------------------------------------------
           DIVIDER
        ---------------------------------------------------- */

        hr {
            border: none;
            border-top: 1px solid var(--line);
            margin: 1.5rem 0;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model(model_path: str):

    model_package = joblib.load(model_path)

    pipeline = model_package["pipeline"]
    high_risk_threshold = model_package["high_risk_threshold"]

    return pipeline, high_risk_threshold


# ============================================================
# MODEL EXPLAINABILITY (SHAP)
# ============================================================
#
# Best-effort feature-level explanation of a single prediction.
# Works with any fitted sklearn Pipeline whose last step is a
# tree-based estimator (e.g. XGBClassifier). If shap is not
# installed, or the pipeline shape isn't what we expect, this
# degrades gracefully — the ML classification itself never
# depends on this panel.

def _split_pipeline(pipeline):
    """Return (preprocessor, final_estimator) from a fitted pipeline."""

    if hasattr(pipeline, "steps") and len(pipeline.steps) > 0:

        from sklearn.pipeline import Pipeline

        if len(pipeline.steps) > 1:
            preprocessor = Pipeline(pipeline.steps[:-1])
        else:
            preprocessor = None

        final_estimator = pipeline.steps[-1][1]

        return preprocessor, final_estimator

    return None, pipeline


def _transform_features(preprocessor, merchant_df):
    """Run the merchant row through preprocessing and recover feature names."""

    if preprocessor is None:
        return merchant_df.to_numpy(), list(merchant_df.columns)

    transformed = preprocessor.transform(merchant_df)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [
            f"feature_{i}" for i in range(transformed.shape[1])
        ]

    return transformed, feature_names


@st.cache_resource(show_spinner=False)
def _get_shap_explainer(_final_estimator):

    import shap

    return shap.TreeExplainer(_final_estimator)


def _pretty_feature_name(name: str) -> str:

    name = re.sub(r"^[a-zA-Z0-9]+__", "", name)
    name = name.replace("_", " ").strip()

    return name[:1].upper() + name[1:] if name else name


def compute_shap_contributions(
    pipeline,
    merchant_df,
    class_index: int = 2,
    top_n: int = 8,
):
    """
    Return the top_n (feature_name, shap_value) pairs for a single
    merchant row, explaining the model's probability for class_index
    (0=Low, 1=Medium, 2=High). Returns None if explanation isn't
    available (shap missing, incompatible pipeline, etc.).
    """

    try:

        import shap  # noqa: F401  (import check only)

        preprocessor, final_estimator = _split_pipeline(pipeline)

        transformed, feature_names = _transform_features(
            preprocessor, merchant_df
        )

        explainer = _get_shap_explainer(final_estimator)

        raw_values = explainer.shap_values(transformed)

        if isinstance(raw_values, list):

            row_values = np.array(raw_values[class_index][0])

        else:

            arr = np.array(raw_values)

            if arr.ndim == 3:

                if arr.shape[0] == transformed.shape[0]:
                    # (n_samples, n_features, n_classes)
                    row_values = arr[0, :, class_index]
                else:
                    # (n_classes, n_samples, n_features)
                    row_values = arr[class_index, 0, :]

            else:

                row_values = arr[0]

        pairs = list(zip(feature_names, row_values))
        pairs.sort(key=lambda pair: abs(pair[1]), reverse=True)

        return pairs[:top_n]

    except Exception:

        return None


def render_shap_bars(contributions) -> None:

    max_abs = max(
        (abs(value) for _, value in contributions),
        default=1.0,
    ) or 1.0

    bars_html = '<div class="prob-bars">'

    for name, value in contributions:

        pretty_name = _pretty_feature_name(name)
        width_pct = (abs(value) / max_abs) * 100
        color = "var(--high)" if value > 0 else "var(--low)"
        sign = "+" if value > 0 else ""

        bars_html += (
            '<div class="shap-bar-row">'
            f'<div class="shap-bar-name">{pretty_name}</div>'
            '<div class="shap-bar-track">'
            f'<div class="shap-bar-fill" '
            f'style="width:{width_pct:.1f}%; background:{color};"></div>'
            '</div>'
            f'<div class="shap-bar-val" style="color:{color};">'
            f'{sign}{value:.3f}</div>'
            '</div>'
        )

    bars_html += '</div>'

    st.markdown(bars_html, unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

def render_business_profile() -> dict:

    st.markdown(
        '<div class="section-label">Business profile</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Basic information about the merchant and verification status.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        business_category = st.selectbox(
            "Business Category",
            BUSINESS_CATEGORIES,
        )

        business_type = st.selectbox(
            "Business Type",
            BUSINESS_TYPES,
        )

    with col2:

        business_age_months = st.number_input(
            "Business Age (months)",
            min_value=0,
            max_value=600,
            value=24,
        )

        kyc_verification_level = st.selectbox(
            "KYC Verification Level",
            KYC_LEVELS,
        )

    with col3:

        registered_state = st.selectbox(
            "Registered State",
            REGISTERED_STATES,
        )

        website_verified = st.radio(
            "Website Verified",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            horizontal=True,
        )

    return {
        "business_category": business_category,
        "business_type": business_type,
        "business_age_months": business_age_months,
        "kyc_verification_level": kyc_verification_level,
        "registered_state": registered_state,
        "website_verified": website_verified,
    }


def render_transaction_behavior() -> dict:

    st.markdown(
        '<div class="section-label">Transaction behavior</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Transaction activity and payment behavior observed for the merchant.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        avg_monthly_txn_volume = st.number_input(
            "Avg. Monthly Transaction Volume",
            min_value=0.0,
            value=500.0,
        )

        txn_success_rate = st.slider(
            "Transaction Success Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=92.0,
        )

    with col2:

        avg_transaction_value = st.number_input(
            "Avg. Transaction Value",
            min_value=0.0,
            value=2500.0,
        )

        refund_ratio = st.slider(
            "Refund Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
        )

    with col3:

        txn_growth_rate_30d = st.number_input(
            "Transaction Growth Rate — 30d (%)",
            value=10.0,
        )

        chargeback_ratio = st.slider(
            "Chargeback Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=2.0,
        )

    night_time_txn_percentage = st.slider(
        "Night-time Transaction Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=15.0,
    )

    return {
        "avg_monthly_txn_volume": avg_monthly_txn_volume,
        "avg_transaction_value": avg_transaction_value,
        "txn_growth_rate_30d": txn_growth_rate_30d,
        "txn_success_rate": txn_success_rate,
        "refund_ratio": refund_ratio,
        "chargeback_ratio": chargeback_ratio,
        "night_time_txn_percentage": night_time_txn_percentage,
    }


def render_settlement_banking() -> dict:

    st.markdown(
        '<div class="section-label">Settlement &amp; banking</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Settlement behavior and linked banking information.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        settlement_delay_requests = st.number_input(
            "Settlement Delay Requests",
            min_value=0,
            value=0,
        )

    with col2:

        bank_account_age_days = st.number_input(
            "Bank Account Age (days)",
            min_value=0,
            value=365,
        )

    with col3:

        reserve_balance_ratio = st.slider(
            "Reserve Balance Ratio (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
        )

    multiple_bank_accounts_flag = st.checkbox(
        "Multiple linked bank accounts",
    )

    return {
        "settlement_delay_requests": settlement_delay_requests,
        "bank_account_age_days": bank_account_age_days,
        "reserve_balance_ratio": reserve_balance_ratio,
        "multiple_bank_accounts_flag": int(
            multiple_bank_accounts_flag
        ),
    }


def render_compliance_risk() -> dict:

    st.markdown(
        '<div class="section-label">Compliance &amp; risk</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Compliance indicators and historical merchant risk information.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        previous_suspension_flag = st.checkbox(
            "Previous suspension",
        )

    with col2:

        complaint_count_90d = st.number_input(
            "Complaints — Last 90 Days",
            min_value=0,
            value=2,
        )

    with col3:

        blacklist_match_score = st.slider(
            "Blacklist Match Score",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
        )

    return {
        "previous_suspension_flag": int(
            previous_suspension_flag
        ),
        "complaint_count_90d": complaint_count_90d,
        "blacklist_match_score": blacklist_match_score,
    }


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(
    low_p: float,
    medium_p: float,
    high_p: float,
    threshold: float,
) -> str:

    if high_p >= threshold:

        return "High"

    elif low_p >= medium_p:

        return "Low"

    else:

        return "Medium"


# ============================================================
# BATCH CSV SCORING
# ============================================================

def generate_csv_template() -> bytes:
    """One-row example CSV showing the columns the batch uploader expects."""

    example_row = {
        "merchant_name": "Example Merchant Pvt Ltd",
        "business_category": BUSINESS_CATEGORIES[0],
        "business_type": BUSINESS_TYPES[0],
        "business_age_months": 24,
        "kyc_verification_level": KYC_LEVELS[0],
        "registered_state": REGISTERED_STATES[0],
        "website_verified": 1,
        "avg_monthly_txn_volume": 500.0,
        "avg_transaction_value": 2500.0,
        "txn_growth_rate_30d": 10.0,
        "txn_success_rate": 92.0,
        "refund_ratio": 5.0,
        "chargeback_ratio": 2.0,
        "night_time_txn_percentage": 15.0,
        "settlement_delay_requests": 0,
        "bank_account_age_days": 365,
        "reserve_balance_ratio": 10.0,
        "multiple_bank_accounts_flag": 0,
        "previous_suspension_flag": 0,
        "complaint_count_90d": 2,
        "blacklist_match_score": 5.0,
    }

    template_df = pd.DataFrame([example_row])

    return template_df.to_csv(index=False).encode("utf-8")


def validate_batch_columns(batch_df) -> list:
    """Return the list of required columns missing from an uploaded CSV."""

    return [
        column
        for column in REQUIRED_COLUMNS
        if column not in batch_df.columns
    ]


def _risk_cell_style(value: str) -> str:

    colors = {
        "Low": "#2fd08a",
        "Medium": "#f2a93b",
        "High": "#ff5468",
    }

    color = colors.get(value)

    return f"color: {color}; font-weight: 600;" if color else ""


def score_batch(pipeline, batch_df, threshold):
    """
    Run every row of batch_df through the pipeline and return a results
    dataframe (original columns + Merchant label + probabilities +
    risk_label), sorted by High-risk probability descending.
    """

    has_names = "merchant_name" in batch_df.columns

    if has_names:
        labels = batch_df["merchant_name"]
    else:
        labels = pd.Series(
            [f"Merchant {i + 1}" for i in range(len(batch_df))]
        )

    scoring_df = batch_df[REQUIRED_COLUMNS].copy()

    probabilities = pipeline.predict_proba(scoring_df)

    results_df = batch_df.copy()

    results_df.insert(0, "Merchant", labels)

    results_df["low_probability"] = probabilities[:, 0]
    results_df["medium_probability"] = probabilities[:, 1]
    results_df["high_probability"] = probabilities[:, 2]

    results_df["risk_label"] = [
        classify_risk(row[0], row[1], row[2], threshold)
        for row in probabilities
    ]

    return results_df.sort_values(
        "high_probability", ascending=False
    ).reset_index(drop=True)


def render_batch_upload(pipeline, threshold) -> None:

    st.markdown(
        '<div class="section-label">Batch merchant scoring</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-description">'
        'Upload a CSV of merchants to score them all in one pass, '
        'using the same model and decision threshold as above.'
        '</div>',
        unsafe_allow_html=True,
    )

    template_col, upload_col = st.columns([1, 2])

    with template_col:

        st.download_button(
            "Download CSV template",
            data=generate_csv_template(),
            file_name="merchant_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader(
        "Merchant CSV",
        type=["csv"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:

        st.caption(
            "Expected columns: " + ", ".join(REQUIRED_COLUMNS) + ". "
            "An optional `merchant_name` column can be included "
            "for labeling rows in the results."
        )

        return

    try:

        batch_df = pd.read_csv(uploaded_file)

    except Exception as e:

        st.error("Could not read the uploaded file as CSV.")

        st.code(str(e))

        return

    if batch_df.empty:

        st.warning("The uploaded CSV has no rows to score.")

        return

    missing_columns = validate_batch_columns(batch_df)

    if missing_columns:

        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join(missing_columns)
        )

        return

    try:

        results_df = score_batch(pipeline, batch_df, threshold)

    except Exception as e:

        st.error(
            "Batch scoring failed. Check that column values match "
            "the expected schema — category spellings, numeric "
            "types, and value ranges."
        )

        st.code(str(e))

        return

    # ----------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------

    counts = results_df["risk_label"].value_counts()

    low_count = int(counts.get("Low", 0))
    medium_count = int(counts.get("Medium", 0))
    high_count = int(counts.get("High", 0))

    st.write("")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Merchants scored", len(results_df))

    with summary_col2:
        st.metric("Low risk", low_count)

    with summary_col3:
        st.metric("Medium risk", medium_count)

    with summary_col4:
        st.metric("High risk", high_count)

    st.write("")

    # ----------------------------------------------------------
    # RESULTS TABLE
    # ----------------------------------------------------------

    display_columns = [
        column
        for column in [
            "Merchant",
            "business_category",
            "risk_label",
            "high_probability",
            "medium_probability",
            "low_probability",
        ]
        if column in results_df.columns
    ]

    styled_table = results_df[display_columns].style.format(
        {
            "high_probability": "{:.2%}",
            "medium_probability": "{:.2%}",
            "low_probability": "{:.2%}",
        }
    )

    if hasattr(styled_table, "map"):
        styled_table = styled_table.map(
            _risk_cell_style, subset=["risk_label"]
        )
    else:
        styled_table = styled_table.applymap(
            _risk_cell_style, subset=["risk_label"]
        )

    st.dataframe(
        styled_table,
        use_container_width=True,
        height=420,
    )

    csv_bytes = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download scored results (CSV)",
        data=csv_bytes,
        file_name="merchant_batch_results.csv",
        mime="text/csv",
        type="primary",
    )

    if high_count > 0:

        st.markdown(
            f"""
            <div class="threshold-card">

            <div class="threshold-title">
            {high_count} merchant(s) flagged High risk
            </div>

            <div class="threshold-text">
            These meet or exceed the {threshold * 100:.0f}%
            High-risk threshold and should route to manual review,
            sorted to the top of the table above.
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# RISK DISPLAY
# ============================================================

RISK_DISPLAY = {

    "Low": {
        "class": "risk-low",
        "recommendation": "Standard monitoring",
        "description": (
            "The predicted High-risk probability is below "
            "the configured threshold."
        ),
    },

    "Medium": {
        "class": "risk-medium",
        "recommendation": "Enhanced monitoring recommended",
        "description": (
            "The merchant does not cross the High-risk threshold, "
            "but Medium risk indicators warrant additional monitoring."
        ),
    },

    "High": {
        "class": "risk-high",
        "recommendation": "Manual review required",
        "description": (
            "The predicted High-risk probability meets or exceeds "
            "the configured decision threshold."
        ),
    },
}


# ============================================================
# AI RISK ANALYST
# ============================================================

def render_ai_risk_analysis(
    merchant_data,
    risk_label,
    low_probability,
    medium_probability,
    high_probability,
    threshold,
) -> None:

    st.divider()

    st.markdown(
        '<div class="section-label">AI investigation layer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ai-container">

        <div class="ai-title">
        AI Risk Analyst
        </div>

        <div class="ai-subtitle">
        Investigator decision support based on the existing ML assessment.
        The AI interprets risk signals and recommends investigation steps.
        It does not change the ML classification.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner(
        "Analyzing merchant risk context..."
    ):

        try:

            result = analyze_merchant_risk(
                merchant_data=merchant_data,
                risk_label=risk_label,
                low_probability=low_probability,
                medium_probability=medium_probability,
                high_probability=high_probability,
                high_risk_threshold=threshold,
            )

            if isinstance(result, dict):

                source = result.get(
                    "source",
                    "local",
                )

                analysis = result.get(
                    "analysis",
                    "",
                )

                signals = result.get(
                    "signals",
                    [],
                )

                recommendations = result.get(
                    "recommendations",
                    [],
                )

                decision = result.get(
                    "decision",
                    None,
                )

                # ------------------------------------------------
                # AI STATUS
                # ------------------------------------------------

                if source == "openai":

                    st.markdown(
                        '<div class="ai-status">'
                        '<span class="status-dot"></span>'
                        'Live AI Analyst'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                else:

                    st.info(
                        "Local AI Risk Analyst active. "
                        "The application is using deterministic "
                        "risk analysis because the live AI service "
                        "is unavailable."
                    )

                # ------------------------------------------------
                # AI ANALYSIS
                # ------------------------------------------------

                if analysis:

                    st.markdown(
                        analysis
                    )

                # ------------------------------------------------
                # ML CONTEXT
                # ------------------------------------------------

                st.markdown(
                    "### ML Decision Context"
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "ML Risk",
                        risk_label,
                    )

                with col2:

                    st.metric(
                        "High-Risk Probability",
                        f"{high_probability * 100:.2f}%",
                    )

                with col3:

                    st.metric(
                        "Decision Threshold",
                        f"{threshold * 100:.0f}%",
                    )

                # ------------------------------------------------
                # DETECTED SIGNALS
                # ------------------------------------------------

                if signals:

                    st.markdown(
                        "### Risk Signals"
                    )

                    signal_cols = st.columns(2)

                    for index, signal in enumerate(
                        signals
                    ):

                        with signal_cols[
                            index % 2
                        ]:

                            st.markdown(
                                f"""
                                <div class="signal-card">
                                <span class="signal-marker">●</span>
                                &nbsp;{signal}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                # ------------------------------------------------
                # RECOMMENDATIONS
                # ------------------------------------------------

                if recommendations:

                    st.markdown(
                        "### Investigation Recommendations"
                    )

                    for index, recommendation in enumerate(
                        recommendations,
                        start=1,
                    ):

                        st.markdown(
                            f"""
                            **{index:02d}** &nbsp;&nbsp;
                            {recommendation}
                            """
                        )

                # ------------------------------------------------
                # HUMAN REVIEW
                # ------------------------------------------------

                if decision:

                    st.markdown(
                        "### Human Review"
                    )

                    st.markdown(
                        f"""
                        <div class="review-card">

                        <div class="review-title">
                        {decision}
                        </div>

                        <div class="review-text">
                        The AI provides decision support only.
                        Final action must be determined by an
                        authorized human reviewer.
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.caption(
                    "The XGBoost model remains the source of the "
                    "risk classification. The AI Risk Analyst "
                    "provides explanation and investigation guidance."
                )

            else:

                st.markdown(
                    result
                )

        except Exception as e:

            st.warning(
                "The AI Risk Analyst is temporarily unavailable. "
                "The ML risk assessment remains valid."
            )

            st.markdown(
                "### ML Investigation Context"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "ML Risk",
                    risk_label,
                )

            with col2:

                st.metric(
                    "High-Risk Probability",
                    f"{high_probability * 100:.2f}%",
                )

            with col3:

                st.metric(
                    "Decision Threshold",
                    f"{threshold * 100:.0f}%",
                )

            st.markdown(
                "### Key Merchant Signals"
            )

            signal_data = {
                "KYC Verification": merchant_data[
                    "kyc_verification_level"
                ],

                "Website Verified": (
                    "Yes"
                    if merchant_data["website_verified"]
                    else "No"
                ),

                "Chargeback Ratio": (
                    f'{merchant_data["chargeback_ratio"]:.2f}%'
                ),

                "Refund Ratio": (
                    f'{merchant_data["refund_ratio"]:.2f}%'
                ),

                "Transaction Success Rate": (
                    f'{merchant_data["txn_success_rate"]:.2f}%'
                ),

                "Night-time Transactions": (
                    f'{merchant_data["night_time_txn_percentage"]:.2f}%'
                ),

                "Previous Suspension": (
                    "Yes"
                    if merchant_data["previous_suspension_flag"]
                    else "No"
                ),

                "Complaints — 90 Days": (
                    merchant_data["complaint_count_90d"]
                ),

                "Blacklist Match Score": (
                    f'{merchant_data["blacklist_match_score"]:.2f}'
                ),

                "Multiple Bank Accounts": (
                    "Yes"
                    if merchant_data["multiple_bank_accounts_flag"]
                    else "No"
                ),
            }

            signal_columns = st.columns(2)

            for index, (
                signal,
                value,
            ) in enumerate(
                signal_data.items()
            ):

                with signal_columns[
                    index % 2
                ]:

                    st.markdown(
                        f"**{signal}**  \n`{value}`"
                    )

            st.markdown(
                "### Recommended Workflow"
            )

            if risk_label == "High":

                st.error(
                    """
                    **Manual review required**

                    1. Re-verify KYC information.
                    2. Review chargeback and refund history.
                    3. Validate linked bank accounts.
                    4. Review previous suspension history.
                    5. Investigate unusual transaction behaviour.
                    """
                )

            elif risk_label == "Medium":

                st.warning(
                    """
                    **Enhanced monitoring recommended**

                    1. Monitor transaction behaviour.
                    2. Review complaints and refund patterns.
                    3. Verify KYC completeness.
                    4. Monitor settlement activity.
                    """
                )

            else:

                st.success(
                    """
                    **Standard monitoring**

                    Continue normal merchant monitoring and
                    periodically reassess risk indicators.
                    """
                )

            with st.expander(
                "Technical information"
            ):

                st.caption(
                    f"AI service status: {type(e).__name__}"
                )


# ============================================================
# RESULT DISPLAY
# ============================================================

def render_result(
    merchant_data,
    merchant_df,
    pipeline,
    low_p,
    medium_p,
    high_p,
    threshold,
) -> None:

    risk_label = classify_risk(
        low_p,
        medium_p,
        high_p,
        threshold,
    )

    display = RISK_DISPLAY[
        risk_label
    ]

    st.divider()

    st.markdown(
        '<div class="section-label">Risk assessment</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # PRIMARY RISK CARD
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="risk-container {display['class']}">

        <div class="risk-label">
        Model Assessment
        </div>

        <div class="risk-value">
        {risk_label} Risk
        </div>

        <div class="risk-description">
        {display['description']}
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    probability_values = [
        ("Low Probability", low_p),
        ("Medium Probability", medium_p),
        ("High Probability", high_p),
    ]

    for column, (
        title,
        value,
    ) in zip(
        [col1, col2, col3],
        probability_values,
    ):

        with column:

            st.markdown(
                f"""
                <div class="prob-card">

                <div class="prob-title">
                {title}
                </div>

                <div class="prob-value">
                {value * 100:.2f}%
                </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    st.write("")

    bar_rows = [
        ("Low", low_p, "var(--low)"),
        ("Medium", medium_p, "var(--medium)"),
        ("High", high_p, "var(--high)"),
    ]

    bar_html = '<div class="prob-bars">'

    for name, value, color in bar_rows:

        pct = value * 100

        bar_html += (
            '<div class="prob-bar-row">'
            f'<div class="prob-bar-name">{name}</div>'
            '<div class="prob-bar-track">'
            f'<div class="prob-bar-fill" '
            f'style="width:{pct:.2f}%; background:{color};"></div>'
            '</div>'
            f'<div class="prob-bar-pct">{pct:.2f}%</div>'
            '</div>'
        )

    bar_html += '</div>'

    st.markdown(
        bar_html,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="threshold-card">

        <div class="threshold-title">
        High-Risk Decision Threshold · {threshold * 100:.0f}%
        </div>

        <div class="threshold-text">
        Merchants with a predicted High-risk probability
        at or above {threshold * 100:.0f}% are classified as High Risk.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # WHY THIS DECISION?
    # --------------------------------------------------------

    with st.expander(
        "Why this decision?"
    ):

        st.write(
            f"**Predicted risk level:** {risk_label}"
        )

        st.write(
            f"**High-risk probability:** "
            f"{high_p * 100:.2f}%"
        )

        st.write(
            f"**Configured threshold:** "
            f"{threshold * 100:.0f}%"
        )

        if high_p >= threshold:

            st.write(
                "The predicted High-risk probability "
                "**met or exceeded** the configured threshold."
            )

        elif risk_label == "Low":

            st.write(
                "The High-risk probability remained below "
                "the threshold and the Low probability was "
                "greater than or equal to the Medium probability."
            )

        else:

            st.write(
                "The High-risk probability remained below "
                "the threshold while the Medium probability "
                "exceeded the Low probability."
            )

    # --------------------------------------------------------
    # FEATURE CONTRIBUTIONS (SHAP)
    # --------------------------------------------------------

    with st.expander(
        "Feature contributions (SHAP)",
        expanded=True,
    ):

        st.caption(
            "How each input pushed this merchant's predicted "
            "High-risk probability up (red) or down (green), "
            "relative to the model's average prediction."
        )

        contributions = compute_shap_contributions(
            pipeline=pipeline,
            merchant_df=merchant_df,
            class_index=2,
            top_n=8,
        )

        if contributions:

            render_shap_bars(contributions)

            st.caption(
                "Values are SHAP contributions in the model's "
                "raw score units, ranked by magnitude for this "
                "specific merchant — not a global feature ranking."
            )

        else:

            st.info(
                "Feature-level explanations need the `shap` "
                "package and a tree-based pipeline. Install it "
                "with `pip install shap` and reassess to enable "
                "this panel."
            )

    # --------------------------------------------------------
    # MODEL PERFORMANCE
    # --------------------------------------------------------

    with st.expander(
        "Model Performance"
    ):

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                f"**Model:** {MODEL_NAME}"
            )

            st.write(
                f"**Test Accuracy:** "
                f"{TEST_ACCURACY * 100:.1f}%"
            )

            st.write(
                f"**High-risk Threshold:** "
                f"{threshold * 100:.0f}%"
            )

        with col2:

            st.write(
                f"**High-risk Precision:** "
                f"{HIGH_RISK_PRECISION * 100:.1f}%"
            )

            st.write(
                f"**High-risk Recall:** "
                f"{HIGH_RISK_RECALL * 100:.1f}%"
            )

            st.write(
                f"**High-risk Confusion (TP/FP/FN):** "
                f"{HIGH_RISK_TP} / "
                f"{HIGH_RISK_FP} / "
                f"{HIGH_RISK_FN}"
            )

        st.caption(
            "The model was trained without merchant_id and "
            "risk_score to prevent target leakage."
        )

    # --------------------------------------------------------
    # AI ANALYST
    # --------------------------------------------------------

    render_ai_risk_analysis(
        merchant_data=merchant_data,
        risk_label=risk_label,
        low_probability=low_p,
        medium_probability=medium_p,
        high_probability=high_p,
        threshold=threshold,
    )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar(
    threshold
) -> None:

    with st.sidebar:

        st.markdown(
            '<div class="sidebar-brand">'
            'Merchant Risk Intelligence'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-caption">'
            'AI-assisted merchant risk assessment '
            'and investigation decision support.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            '<div class="sidebar-item">'
            '<div class="sidebar-label">Risk Model</div>'
            '<div class="sidebar-value">XGBoost</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-item">'
            '<div class="sidebar-label">AI Layer</div>'
            '<div class="sidebar-value">AI Risk Analyst</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-item">'
            '<div class="sidebar-label">Classification</div>'
            '<div class="sidebar-value">'
            'Low / Medium / High'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="sidebar-item">'
            '<div class="sidebar-label">High-Risk Threshold</div>'
            f'<div class="sidebar-value">'
            f'{threshold * 100:.0f}%'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown(
            """
            <div class="sidebar-caption">
            The AI analyst provides explanation,
            investigation guidance, and contextual
            analysis. Final risk decisions remain
            with authorized human reviewers.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    inject_custom_css()

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        pipeline, high_risk_threshold = load_model(
            MODEL_PATH
        )

    except FileNotFoundError:

        st.error(
            f"Model file not found at `{MODEL_PATH}`."
        )

        st.stop()

    except Exception as e:

        st.error(
            "Could not load the trained model."
        )

        st.code(
            str(e)
        )

        st.stop()

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    render_sidebar(
        high_risk_threshold
    )

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    header_col1, header_col2 = st.columns(
        [5, 1]
    )

    with header_col1:

        st.markdown(
            '<div class="app-header">'
            '<div class="app-eyebrow">RISK-01 · MERCHANT SCREENING</div>'
            '<div class="app-title">'
            'Merchant Risk Intelligence'
            '</div>'
            '<div class="app-subtitle">'
            'AI-assisted merchant risk assessment and '
            'investigation decision support'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    with header_col2:

        st.markdown(
            '<div style="text-align:right; padding-top:0.6rem;">'
            '<div class="system-status">'
            '<span class="status-dot"></span>'
            'MODEL ONLINE'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    tab_single, tab_batch = st.tabs(
        ["Single merchant", "Batch upload"]
    )

    # --------------------------------------------------------
    # SINGLE MERCHANT TAB
    # --------------------------------------------------------

    with tab_single:

        business_inputs = render_business_profile()

        st.write("")

        transaction_inputs = render_transaction_behavior()

        st.write("")

        settlement_inputs = render_settlement_banking()

        st.write("")

        compliance_inputs = render_compliance_risk()

        merchant_data = {
            **business_inputs,
            **transaction_inputs,
            **settlement_inputs,
            **compliance_inputs,
        }

        # ----------------------------------------------------
        # ASSESS
        # ----------------------------------------------------

        st.divider()

        _, center, _ = st.columns(
            [1, 2, 1]
        )

        with center:

            assess_clicked = st.button(
                "Assess Merchant Risk",
                use_container_width=True,
                type="primary",
            )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if assess_clicked:

            merchant_df = pd.DataFrame(
                [merchant_data]
            )

            try:

                probabilities = pipeline.predict_proba(
                    merchant_df
                )[0]

            except Exception as e:

                st.error(
                    "Prediction failed. The supplied inputs "
                    "may not match the model schema."
                )

                st.code(
                    str(e)
                )

                st.stop()

            low_probability = probabilities[0]
            medium_probability = probabilities[1]
            high_probability = probabilities[2]

            render_result(
                merchant_data=merchant_data,
                merchant_df=merchant_df,
                pipeline=pipeline,
                low_p=low_probability,
                medium_p=medium_probability,
                high_p=high_probability,
                threshold=high_risk_threshold,
            )

    # --------------------------------------------------------
    # BATCH UPLOAD TAB
    # --------------------------------------------------------

    with tab_batch:

        render_batch_upload(
            pipeline=pipeline,
            threshold=high_risk_threshold,
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()