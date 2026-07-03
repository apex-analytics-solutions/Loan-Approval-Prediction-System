import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
from pdf_generator import generate_pdf_report
from predict_engine import predict as run_prediction, validate_feature_alignment
from utils import (
    setup_logging, load_artifacts_cached, ARTIFACTS_PATH,
    MODEL_COMPARISON_CSV, DASHBOARD_DATA_PATH, DATASET_PATH,
    OUTPUTS_DIR, VERSION,
)
import uuid
import re
import hashlib
import base64
import asyncio
import datetime
import sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

log = setup_logging("app", log_to_file=False)

# =========================
# SESSION STATE — single initialization block
# =========================
_DEFAULT_STATE = {
    "chat_history": [],
    "last_applicant_data": None,
    "last_prediction": None,
    "last_probability": None,
    "application_id": f"APP-{uuid.uuid4().hex[:8].upper()}",
    "raw_pred": None,
    "raw_proba": None,
    "risk_level": None,
    "risk_color": None,
    "risk_emoji": None,
    "confidence": None,
    "pdf_ready": False,
    "pdf_bytes": None,
    "pdf_file_name": None,
    "prediction_counter": 0,
    "model_name": None,
    "prediction_timestamp": None,
    "prediction_input_hash": None,
}
for _key, _default in _DEFAULT_STATE.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Session state integrity: reset corrupted values to defaults ──
_STATE_TYPES = {
    "raw_pred": (int, type(None)),
    "raw_proba": (float, int, type(None)),
    "risk_level": (str, type(None)),
    "risk_color": (str, type(None)),
    "risk_emoji": (str, type(None)),
    "confidence": (str, type(None)),
    "prediction_counter": (int,),
    "pdf_ready": (bool,),
    "chat_history": (list,),
    "last_applicant_data": (dict, type(None)),
    "last_prediction": (str, type(None)),
    "last_probability": (float, int, type(None)),
}
for _sk, _expected_types in _STATE_TYPES.items():
    if _sk in st.session_state and not isinstance(st.session_state[_sk], _expected_types):
        log.warning(
            "Session state '%s' has unexpected type %s, resetting to default",
            _sk, type(st.session_state[_sk]).__name__,
        )
        st.session_state[_sk] = _DEFAULT_STATE[_sk]


def _hash_inputs(raw: dict) -> str:
    """Deterministic fingerprint of the prediction inputs."""
    canonical = str(sorted(raw.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _chart_theme():
    """Return Plotly-compatible color dict that adapts to the current dark/light toggle."""
    dark = st.session_state.get("dark_mode", True)
    if dark:
        return {
            "bg": "rgba(0,0,0,0)",
            "text": "#e2e8f0",
            "title": "#f8fafc",
            "axis": "#94a3b8",
            "grid": "rgba(148,163,184,0.12)",
            "legend": "#cbd5e1",
            "muted": "#64748b",
            "surface": "#182235",
        }
    return {
        "bg": "rgba(0,0,0,0)",
        "text": "#334155",
        "title": "#0f172a",
        "axis": "#64748b",
        "grid": "rgba(226,232,240,0.8)",
        "legend": "#475569",
        "muted": "#94a3b8",
        "surface": "#ffffff",
    }


CB_PALETTE = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4", "#ec4899"]


MENU_TOPICS = (
    "• **Loan status** — current decision and probability\n"
    "• **Why approved / rejected** — factors behind the decision\n"
    "• **How to improve** — actionable steps to raise approval chances\n"
    "• **Explain credit score** — what your score means\n"
    "• **Explain debt ratio** — how your debt-to-income ratio is evaluated\n"
    "• **Explain probability** — how the approval probability works\n"
    "• **Explain risk level** — what your risk category means\n"
    "• **Financial profile** — full summary of strengths and risks"
)

UNSTABLE_EMPLOYMENT = {"unemployed", "temporary", "contract"}


def loan_chatbot_response(user_input, applicant_data, prediction, probability):
    """Return a professional rule-based assistant response using applicant risk context."""
    question = (user_input or "").strip().lower()

    # --- Greeting / help (works without prediction) ---
    if question in {"hi", "hello", "hey", "help", "menu", "?"}:
        return (
            f"Welcome to the AI Financial Assistant. I can help you with:\n\n"
            f"{MENU_TOPICS}\n\n"
            "Please run a prediction first, then ask any question above."
        )

    if not applicant_data or prediction is None or probability is None:
        return (
            "I can assist with your loan analysis once a prediction is generated. "
            "Please click **Predict Loan Status** first, then ask me anything."
        )

    # --- Extract applicant values (safe conversion) ---
    try:
        credit_score = float(applicant_data.get("Credit Score", 0))
        debt_to_income_ratio = float(applicant_data.get("Debt To Income Ratio", 0))
        annual_income = float(applicant_data.get("Annual Income", 0))
        monthly_income = float(applicant_data.get("Monthly Income", 0))
        loan_amount = float(applicant_data.get("Loan Amount", 0))
        interest_rate = float(applicant_data.get("Interest Rate", 0))
        employment = str(applicant_data.get("Employment Status", "Unknown"))
        employment_lower = employment.lower()
        education = str(applicant_data.get("Education Level", "Unknown"))
        probability_pct = float(probability) * 100
        loan_to_income = (loan_amount / annual_income) if annual_income > 0 else 999
        debt_pct = debt_to_income_ratio * 100
    except (TypeError, ValueError, ZeroDivisionError):
        return (
            "I encountered an issue reading your financial data. "
            "Please run a new prediction and try again."
        )

    # --- Build risk flags and strengths ---
    risk_flags = []
    strengths = []

    if credit_score < 580:
        risk_flags.append(f"credit score ({credit_score:.0f}) is significantly below the 620 bank threshold")
    elif credit_score < 620:
        risk_flags.append(f"credit score ({credit_score:.0f}) is below the preferred 620 threshold")
    elif credit_score >= 720:
        strengths.append(f"credit score ({credit_score:.0f}) reflects strong repayment reliability")
    else:
        strengths.append(f"credit score ({credit_score:.0f}) is within acceptable range")

    if debt_to_income_ratio > 0.50:
        risk_flags.append(f"debt-to-income ratio ({debt_pct:.1f}%) is critically elevated above 50%")
    elif debt_to_income_ratio > 0.43:
        risk_flags.append(f"debt-to-income ratio ({debt_pct:.1f}%) exceeds the 43% guideline")
    elif debt_to_income_ratio <= 0.30:
        strengths.append(f"debt-to-income ratio ({debt_pct:.1f}%) is healthy and well-controlled")
    else:
        strengths.append(f"debt-to-income ratio ({debt_pct:.1f}%) is within acceptable range")

    if annual_income < 30000:
        risk_flags.append(f"annual income (${annual_income:,.0f}) may be insufficient for the requested loan")
    elif annual_income >= 70000:
        strengths.append(f"annual income (${annual_income:,.0f}) demonstrates strong repayment capacity")
    else:
        strengths.append(f"annual income (${annual_income:,.0f}) is within a moderate range")

    if loan_to_income > 0.6:
        risk_flags.append(f"loan amount (${loan_amount:,.0f}) is high relative to annual income ({loan_to_income:.1%} ratio)")
    elif loan_to_income <= 0.35:
        strengths.append(f"loan amount (${loan_amount:,.0f}) is proportionate to income ({loan_to_income:.1%} ratio)")

    if employment_lower in UNSTABLE_EMPLOYMENT:
        risk_flags.append(f"employment status ({employment}) may signal income instability")
    else:
        strengths.append(f"employment status ({employment}) supports credit stability")

    if interest_rate > 18.0:
        risk_flags.append(f"interest rate ({interest_rate:.1f}%) is high, indicating elevated lender risk assessment")
    elif interest_rate <= 8.0:
        strengths.append(f"interest rate ({interest_rate:.1f}%) is favorable")

    # --- Determine risk label (thresholds match predict_engine._classify_risk) ---
    if probability_pct >= 75:
        risk_label = "LOW"
    elif probability_pct >= 50:
        risk_label = "MEDIUM"
    else:
        risk_label = "HIGH"

    # --- Intent matching ---

    # Loan status
    if "status" in question or "decision" in question:
        return (
            f"**Current Loan Decision:** {prediction}\n\n"
            f"**Approval Probability:** {probability_pct:.2f}%\n\n"
            f"**Risk Level:** {risk_label}\n\n"
            f"The AI model evaluated your complete financial profile including credit score "
            f"({credit_score:.0f}), income (${annual_income:,.0f}), debt ratio ({debt_pct:.1f}%), "
            f"and employment status ({employment}) to reach this decision."
        )

    # Why rejected
    if "why" in question and ("reject" in question or "denied" in question or "not approved" in question):
        if str(prediction).lower() == "rejected":
            flag_text = "\n".join(f"  {i+1}. {f.capitalize()}" for i, f in enumerate(risk_flags))
            response = f"**Application Status:** Rejected (Probability: {probability_pct:.2f}%)\n\n"
            if risk_flags:
                response += f"The application was declined due to the following risk factors:\n\n{flag_text}\n\n"
            else:
                response += "The model detected elevated overall risk based on the combination of your financial indicators.\n\n"
            response += "Type **how to improve** to receive a personalized improvement plan."
            return response
        return (
            f"Your application is currently **Approved** with {probability_pct:.2f}% probability. "
            "If you want to understand risk factors that could affect future applications, "
            "ask me about **risk level** or **financial profile**."
        )

    # Why approved
    if "why" in question and ("approved" in question or "accepted" in question):
        if str(prediction).lower() == "approved":
            strength_text = "\n".join(f"  {i+1}. {s.capitalize()}" for i, s in enumerate(strengths))
            response = f"**Application Status:** Approved (Probability: {probability_pct:.2f}%)\n\n"
            if strengths:
                response += f"The approval is supported by the following strengths:\n\n{strength_text}"
            else:
                response += "The model found your overall financial profile acceptable based on combined indicators."
            return response
        return (
            f"Your application is currently **Rejected**. "
            "I can provide a personalized improvement plan — ask me **how to improve**."
        )

    # How to improve
    if "improve" in question or "how can i" in question or "tips" in question or "advice" in question:
        steps = []
        if credit_score < 720:
            gap = 720 - credit_score
            steps.append(f"Raise your credit score by ~{gap:.0f} points to reach 720+ through consistent on-time payments and reducing credit utilization below 30%")
        if debt_to_income_ratio > 0.35:
            steps.append(f"Reduce your debt-to-income ratio from {debt_pct:.1f}% to below 35% by paying down existing obligations")
        if annual_income < 50000:
            steps.append("Increase verified annual income through career advancement, additional certifications, or supplemental income sources")
        if loan_to_income > 0.5 and annual_income > 0:
            suggested = annual_income * 0.35
            steps.append(f"Consider requesting a lower loan amount closer to ${suggested:,.0f} (35% of your annual income)")
        if employment_lower in UNSTABLE_EMPLOYMENT:
            steps.append("Secure stable full-time or permanent employment before reapplying")
        if interest_rate > 15.0:
            steps.append("Improving your credit profile will help qualify for a lower interest rate")
        if not steps:
            steps.append("Your profile is already strong — maintain current financial discipline and avoid taking on new debt before the loan closes")

        step_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        return f"**Personalized Improvement Plan:**\n\n{step_text}"

    # Explain credit score
    if "credit" in question and "score" in question:
        if credit_score >= 750:
            assessment = "Excellent. You are in the top tier of borrowers. Lenders offer the best rates and terms at this level."
        elif credit_score >= 720:
            assessment = "Very Good. This score qualifies you for competitive interest rates and favorable loan terms."
        elif credit_score >= 670:
            assessment = "Good. Most lenders consider this acceptable, though premium rates may require a higher score."
        elif credit_score >= 620:
            assessment = "Fair. You meet minimum thresholds for many lenders, but terms may be less favorable."
        elif credit_score >= 580:
            assessment = "Below Average. Many traditional lenders may decline at this level. Focused credit repair is recommended."
        else:
            assessment = "Poor. Significant credit repair is needed before most lenders will approve a loan application."
        return (
            f"**Your Credit Score:** {credit_score:.0f} / 850\n\n"
            f"**Assessment:** {assessment}\n\n"
            "**Score Ranges:**\n"
            "  • 750-850: Excellent\n"
            "  • 720-749: Very Good\n"
            "  • 670-719: Good\n"
            "  • 620-669: Fair\n"
            "  • 580-619: Below Average\n"
            "  • 300-579: Poor\n\n"
            "Credit scores are influenced by payment history, credit utilization, "
            "length of credit history, credit mix, and recent inquiries."
        )

    # Explain debt-to-income ratio
    if "debt" in question and ("ratio" in question or "income" in question):
        if debt_to_income_ratio <= 0.20:
            assessment = "Excellent. Your debt obligations are very low relative to income, indicating strong financial health."
        elif debt_to_income_ratio <= 0.35:
            assessment = "Good. Your debt level is manageable and within preferred lending guidelines."
        elif debt_to_income_ratio <= 0.43:
            assessment = "Acceptable. You are near the upper limit that most lenders allow for qualified mortgages."
        elif debt_to_income_ratio <= 0.50:
            assessment = "Elevated. Most banks prefer below 43%. Reducing existing debt is recommended."
        else:
            assessment = "Critical. This ratio significantly exceeds lending guidelines and is a primary rejection factor."
        return (
            f"**Your Debt-to-Income Ratio:** {debt_pct:.1f}%\n\n"
            f"**Assessment:** {assessment}\n\n"
            "**Industry Guidelines:**\n"
            "  • Below 20%: Excellent\n"
            "  • 20%-35%: Good\n"
            "  • 36%-43%: Acceptable\n"
            "  • 44%-50%: Elevated Risk\n"
            "  • Above 50%: Critical\n\n"
            f"With a monthly income of ${monthly_income:,.0f}, your total monthly debt obligations "
            f"should ideally stay below ${monthly_income * 0.35:,.0f} (35% threshold)."
        )

    # Explain probability
    if "probability" in question or "chance" in question or "likelihood" in question:
        if probability_pct >= 80:
            outlook = "Strong approval outlook. The model is highly confident in a positive outcome."
        elif probability_pct >= 60:
            outlook = "Moderate approval outlook. The profile shows more positive signals than negative."
        elif probability_pct >= 40:
            outlook = "Borderline. The model finds roughly equal positive and negative signals."
        else:
            outlook = "Weak approval outlook. Significant profile improvements are needed."
        return (
            f"**Approval Probability:** {probability_pct:.2f}%\n\n"
            f"**Interpretation:** {outlook}\n\n"
            "The probability is calculated by the trained ML model based on your complete "
            "financial profile. It represents the likelihood that a borrower with your exact "
            "characteristics would successfully repay the loan.\n\n"
            "**Key factors driving this probability:**\n"
            f"  • Credit Score: {credit_score:.0f}\n"
            f"  • Debt-to-Income: {debt_pct:.1f}%\n"
            f"  • Annual Income: ${annual_income:,.0f}\n"
            f"  • Loan Amount: ${loan_amount:,.0f}\n"
            f"  • Employment: {employment}"
        )

    # Explain risk level
    if "risk" in question and ("level" in question or "category" in question or "rating" in question):
        if risk_label == "LOW":
            detail = (
                "Your profile presents minimal default risk. Strong financial indicators across "
                "credit, income, and debt management place you in the most favorable category."
            )
        elif risk_label == "MEDIUM":
            detail = (
                "Your profile shows moderate risk. While several indicators are acceptable, "
                "there are areas that could be strengthened to move into the low-risk category."
            )
        else:
            detail = (
                "Your profile shows elevated default risk. Multiple financial indicators fall "
                "outside preferred lending thresholds. A focused improvement plan is recommended."
            )
        return (
            f"**Your Risk Level:** {risk_label}\n\n"
            f"**Detail:** {detail}\n\n"
            "**Risk Categories:**\n"
            "  • LOW (probability >= 75%): Strong profile, favorable terms likely\n"
            "  • MEDIUM (probability 50-74%): Acceptable profile, standard terms\n"
            "  • HIGH (probability < 50%): Elevated risk, improvement needed\n\n"
            f"Your current approval probability of {probability_pct:.2f}% places you in the "
            f"**{risk_label}** risk band."
        )

    # Risk factors (general)
    if "factor" in question or "risk" in question:
        all_factors = []
        if risk_flags:
            all_factors.append("**Risk Factors:**")
            for f in risk_flags:
                all_factors.append(f"  • {f.capitalize()}")
        if strengths:
            all_factors.append("\n**Strengths:**")
            for s in strengths:
                all_factors.append(f"  • {s.capitalize()}")
        return "\n".join(all_factors) if all_factors else "No significant risk factors or strengths identified."

    # Financial profile summary
    if "financial" in question or "profile" in question or "summary" in question:
        strength_lines = "\n".join(f"  • {s.capitalize()}" for s in strengths) if strengths else "  • No significant strengths identified"
        risk_lines = "\n".join(f"  • {f.capitalize()}" for f in risk_flags) if risk_flags else "  • No major risk flags detected"
        return (
            f"**Financial Profile Summary**\n\n"
            f"**Decision:** {prediction} | **Probability:** {probability_pct:.2f}% | **Risk:** {risk_label}\n\n"
            f"**Strengths:**\n{strength_lines}\n\n"
            f"**Watch Areas:**\n{risk_lines}\n\n"
            f"**Key Metrics:**\n"
            f"  • Credit Score: {credit_score:.0f} / 850\n"
            f"  • Debt-to-Income: {debt_pct:.1f}%\n"
            f"  • Annual Income: ${annual_income:,.0f}\n"
            f"  • Monthly Income: ${monthly_income:,.0f}\n"
            f"  • Loan Amount: ${loan_amount:,.0f}\n"
            f"  • Interest Rate: {interest_rate:.1f}%\n"
            f"  • Employment: {employment}\n"
            f"  • Education: {education}"
        )

    # Fallback
    return (
        f"I can help you with the following topics:\n\n"
        f"{MENU_TOPICS}\n\n"
        "Try asking one of the questions above."
    )


def _build_pdf_filename(app_id: str, name: str) -> str:
    safe_name = re.sub(r'[^\w\s-]', '', name.strip()).replace(' ', '_')
    safe_id = re.sub(r'[^\w-]', '', app_id.strip())
    if safe_name:
        return f"{safe_id}_{safe_name}_Loan_Assessment_Report.pdf"
    return f"{safe_id}_Loan_Assessment_Report.pdf"


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Loan Intelligence System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# THEME TOGGLE (JavaScript — no page rerun)
# =========================
with st.sidebar:
    dark_mode = st.toggle("🌙 Dark Mode", value=True, key="dark_mode")

# =========================
# PREMIUM THEME-ADAPTIVE UI
# =========================
st.markdown("""
<style>
/* ============================================================
   FINTECH THEME SYSTEM — Dark default, light via Python toggle
   Organized: Variables > Layout > Typography > Sidebar >
   Inputs > Buttons > Cards > Charts > Tables > Sections >
   Popups > Footer > Modals > Expanders > Messages > Responsive
   ============================================================ */

/* ──────────── 1. CSS CUSTOM PROPERTIES ──────────── */
:root {
    --lp-bg: radial-gradient(circle at 15% 10%, #1a2740 0%, #0b1220 48%, #0a1120 100%);
    --lp-sidebar: linear-gradient(180deg, #0f172a 0%, #101a30 100%);
    --lp-surface: #182235;
    --lp-surface-grad: linear-gradient(180deg, #182235 0%, #1f2d45 100%);
    --lp-text: #f8fafc;
    --lp-text-2: #cbd5e1;
    --lp-text-3: #94a3b8;
    --lp-border: rgba(148, 163, 184, 0.25);
    --lp-input-bg: #111c33;
    --lp-input-text: #e2e8f0;
    --lp-input-border: rgba(148, 163, 184, 0.35);
    --lp-header-bg: rgba(11, 18, 32, 0.75);
    --lp-card-shadow: 0 8px 20px rgba(0, 0, 0, 0.24);
    --lp-chart-border: rgba(148, 163, 184, 0.15);
    --lp-hr: rgba(148, 163, 184, 0.18);
    --lp-panel-bg: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    --lp-panel-border: rgba(47, 124, 255, 0.25);
    --lp-panel-shadow: 0 6px 24px rgba(0, 0, 0, 0.22);
    --lp-panel-cell: rgba(255, 255, 255, 0.06);
    --lp-rec-bg: linear-gradient(135deg, #0f4c81 0%, #1a365d 100%);
    --lp-accent: #38bdf8;
    --lp-radius: 14px;
    --lp-radius-sm: 10px;
    --lp-spacing-xs: 0.35rem;
    --lp-spacing-sm: 0.5rem;
    --lp-spacing-md: 1rem;
    --lp-spacing-lg: 1.5rem;
    --lp-spacing-xl: 2rem;
}

/* ──────────── 2. MAIN LAYOUT ──────────── */

[data-testid="stAppViewContainer"] {
    background: var(--lp-bg);
    color: var(--lp-text);
}

.stApp {
    color: var(--lp-text);
}

[data-testid="stHeader"] {
    background: var(--lp-header-bg);
    backdrop-filter: blur(6px);
}

[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] span {
    color: #FFFFFF !important;
    opacity: 1 !important; /* Always visible */
}

[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapseButton"] button:active,
[data-testid="stExpandSidebarButton"]:hover,
[data-testid="stExpandSidebarButton"]:active {
    color: #FFFFFF !important;
    opacity: 0.85 !important;
}

.main .block-container {
    max-width: 1200px;
    padding: 4rem 1.5rem 2.5rem;
}

/* ──────────── 3. TYPOGRAPHY ──────────── */
h1 {
    font-size: 1.75rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    line-height: 1.25 !important;
    color: var(--lp-text) !important;
    margin-bottom: var(--lp-spacing-md) !important;
}

h2 {
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px !important;
    line-height: 1.3 !important;
    color: var(--lp-text) !important;
    margin-bottom: var(--lp-spacing-sm) !important;
}

h3 {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.2px !important;
    line-height: 1.35 !important;
    color: var(--lp-text) !important;
    margin-bottom: var(--lp-spacing-xs) !important;
}

h4, h5, h6 {
    font-weight: 600 !important;
    color: var(--lp-text) !important;
}

p, li, label,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stText"] {
    color: var(--lp-text) !important;
    line-height: 1.6 !important;
}

[data-testid="stCaptionContainer"],
small,
[data-testid="stMetricDelta"] {
    color: var(--lp-text-3) !important;
}

/* ──────────── 4. SIDEBAR ──────────── */
[data-testid="stSidebar"] {
    background: var(--lp-sidebar);
    border-right: 1px solid var(--lp-border);
}

[data-testid="stSidebar"] > div:first-child {
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding-bottom: var(--lp-spacing-lg) !important;
}


[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: var(--lp-text) !important;
}

[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] .stSlider {
    background: var(--lp-input-bg);
    border-radius: var(--lp-radius-sm);
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] [role="option"] {
    color: var(--lp-input-text) !important;
}

[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder {
    color: var(--lp-text-3) !important;
}

[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stDateInput input,
[data-testid="stSidebar"] .stTimeInput input {
    background: var(--lp-input-bg) !important;
    border: 1px solid var(--lp-input-border) !important;
    border-radius: var(--lp-radius-sm) !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] div[role="combobox"] {
    background: var(--lp-input-bg) !important;
    border: 1px solid var(--lp-input-border) !important;
    border-radius: var(--lp-radius-sm) !important;
}

.sidebar-section-header {
    background: linear-gradient(90deg, rgba(0, 194, 255, 0.15), rgba(47, 124, 255, 0.08));
    border-left: 3px solid #2f7cff;
    border-radius: 0 8px 8px 0;
    padding: var(--lp-spacing-sm) 0.75rem;
    margin: var(--lp-spacing-md) 0 0.6rem 0;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.3px;
    color: var(--lp-text) !important;
}

.sidebar-app-id {
    background: var(--lp-surface);
    border: 1px solid var(--lp-border);
    border-radius: var(--lp-radius-sm);
    padding: 0.6rem 0.75rem;
    margin-bottom: 0.8rem;
}

.sidebar-app-id .app-id-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    color: var(--lp-text-3) !important;
    margin-bottom: 0.2rem;
}

.sidebar-app-id .app-id-value {
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    color: var(--lp-accent) !important;
}

.sidebar-divider {
    border: 0;
    border-top: 1px solid var(--lp-border);
    margin: var(--lp-spacing-sm) 0;
}

.validation-warn {
    font-size: 0.75rem;
    color: #f59e0b !important;
    margin-top: -0.4rem;
    margin-bottom: 0.3rem;
    padding-left: 0.2rem;
}

.sidebar-form-title {
    text-align: center;
    margin-bottom: 0.6rem;
}

.sidebar-form-title .bank-icon {
    font-size: 1.4rem;
    display: block;
    margin-bottom: 0.15rem;
}

.sidebar-form-title .form-heading {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 0.4px;
    color: var(--lp-text) !important;
    margin: 0;
}

.sidebar-form-title .bank-name {
    font-size: 0.73rem;
    font-weight: 500;
    color: var(--lp-text-3) !important;
    margin: 0.1rem 0 0 0;
}

/* ──────────── 5. INPUTS (MAIN AREA) ──────────── */
[data-baseweb="input"],
[data-baseweb="select"],
.stNumberInput,
.stSelectbox,
.stSlider {
    color: var(--lp-text) !important;
}

/* ──────────── 6. BUTTONS ──────────── */
.stButton > button {
    width: 100%;
    min-height: 48px;
    border: 0;
    border-radius: 12px;
    background: linear-gradient(90deg, #00c2ff, #2f7cff);
    color: #ffffff !important;
    font-weight: 700;
    font-size: 0.98rem;
    letter-spacing: 0.2px;
    box-shadow: 0 10px 22px rgba(0, 111, 255, 0.35);
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.stButton > button:hover {
    transform: translateY(-1px);
    filter: brightness(1.04);
    box-shadow: 0 14px 30px rgba(0, 111, 255, 0.45);
}

.stButton > button:focus {
    outline: 2px solid rgba(148, 163, 184, 0.5);
    outline-offset: 2px;
}

.stDownloadButton > button {
    background: linear-gradient(90deg, #10b981, #059669) !important;
    border: 0 !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
    transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
}

.stDownloadButton > button:hover {
    filter: brightness(1.08) !important;
    box-shadow: 0 12px 28px rgba(16, 185, 129, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ──────────── 7. METRIC CARDS ──────────── */
.metric-card {
    background: var(--lp-surface-grad);
    border: 1px solid var(--lp-border);
    border-radius: var(--lp-radius);
    padding: var(--lp-spacing-md) var(--lp-spacing-md) 0.9rem;
    min-height: 122px;
    box-shadow: var(--lp-card-shadow);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
}

.metric-label {
    color: var(--lp-text-2) !important;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    margin-bottom: var(--lp-spacing-xs);
}

.metric-value {
    color: var(--lp-text) !important;
    font-size: 1.45rem;
    font-weight: 800;
    line-height: 1.2;
}

/* ──────────── 8. KPI CARDS ──────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: var(--lp-spacing-md);
    margin-bottom: var(--lp-spacing-lg);
}

.kpi-card {
    background: var(--lp-surface-grad);
    border: 1px solid var(--lp-border);
    border-radius: var(--lp-radius);
    padding: 1.1rem 1.2rem;
    box-shadow: var(--lp-card-shadow);
    position: relative;
    overflow: hidden;
    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.3);
}

.kpi-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 3px;
    border-radius: var(--lp-radius) var(--lp-radius) 0 0;
}

.kpi-card.kpi-blue::after { background: linear-gradient(90deg, #2f7cff, #60a5fa); }
.kpi-card.kpi-green::after { background: linear-gradient(90deg, #10b981, #34d399); }
.kpi-card.kpi-purple::after { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.kpi-card.kpi-cyan::after { background: linear-gradient(90deg, #06b6d4, #22d3ee); }
.kpi-card.kpi-amber::after { background: linear-gradient(90deg, #f59e0b, #fbbf24); }

.kpi-card .kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: var(--lp-text-3) !important;
    margin-bottom: 0.4rem;
}

.kpi-card .kpi-value {
    font-size: 1.35rem;
    font-weight: 800;
    color: var(--lp-text) !important;
    line-height: 1.2;
}

.kpi-card .kpi-sub {
    font-size: 0.72rem;
    color: var(--lp-text-3) !important;
    margin-top: 0.3rem;
}

/* ──────────── 9. CHART CONTAINERS ──────────── */
[data-testid="stPlotlyChart"] {
    background: var(--lp-surface);
    border-radius: var(--lp-radius);
    padding: var(--lp-spacing-sm);
    border: 1px solid var(--lp-chart-border);
    box-shadow: var(--lp-card-shadow);
    margin-bottom: var(--lp-spacing-md);
    overflow: hidden;
}

/* ──────────── 10. DATA TABLES ──────────── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: var(--lp-radius) !important;
    overflow: hidden;
}

[data-testid="stDataFrame"] > div {
    border-radius: var(--lp-radius) !important;
    border: 1px solid var(--lp-chart-border) !important;
}

/* ──────────── 11. SECTION HEADERS ──────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: var(--lp-spacing-sm);
    margin: var(--lp-spacing-xl) 0 var(--lp-spacing-md) 0;
    padding-bottom: 0.7rem;
    border-bottom: 1px solid var(--lp-hr);
}

.section-header .section-icon {
    font-size: 1.3rem;
    line-height: 1;
}

.section-header .section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--lp-text) !important;
    margin: 0;
    letter-spacing: -0.2px;
}

.section-header .section-badge {
    margin-left: auto;
    background: rgba(47, 124, 255, 0.1);
    border: 1px solid rgba(47, 124, 255, 0.2);
    border-radius: 16px;
    padding: 3px 12px;
    font-size: 0.68rem;
    font-weight: 600;
    color: #60a5fa;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ──────────── 12. HERO BANNER ──────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1a2740 100%);
    border-radius: 18px;
    padding: 2.2rem 2.4rem;
    margin-bottom: var(--lp-spacing-xl);
    border: 1px solid rgba(47, 124, 255, 0.2);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 60%;
    height: 200%;
    background: radial-gradient(ellipse, rgba(47, 124, 255, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-banner .hero-badge {
    display: inline-block;
    background: rgba(47, 124, 255, 0.15);
    border: 1px solid rgba(47, 124, 255, 0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 0.8rem;
}

.hero-banner .hero-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

.hero-banner .hero-subtitle {
    font-size: 1rem;
    font-weight: 500;
    color: #94a3b8;
    margin: 0 0 0.6rem 0;
}

.hero-banner .hero-desc {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0;
    max-width: 600px;
    line-height: 1.6;
}

/* ──────────── 13. POPUPS & DROPDOWNS ──────────── */
[data-baseweb="popover"] > div {
    background: var(--lp-surface) !important;
    border: 1px solid var(--lp-border) !important;
    border-radius: var(--lp-radius-sm) !important;
    box-shadow: var(--lp-card-shadow) !important;
}

[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: var(--lp-surface) !important;
}

[data-baseweb="popover"] li,
[data-baseweb="popover"] a,
[data-baseweb="popover"] span,
[data-baseweb="popover"] p,
[data-baseweb="popover"] button,
[data-baseweb="popover"] div,
[data-baseweb="menu"] li,
[data-baseweb="menu"] a,
[data-baseweb="menu"] span {
    color: var(--lp-text) !important;
}

[data-baseweb="popover"] li:hover,
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="menu"] li:hover {
    background: var(--lp-input-bg) !important;
}

[role="listbox"] {
    background: var(--lp-surface) !important;
}

[role="listbox"] [role="option"] {
    color: var(--lp-text) !important;
    background: var(--lp-surface) !important;
}

[role="listbox"] [role="option"]:hover,
[role="listbox"] [role="option"][aria-selected="true"] {
    background: var(--lp-input-bg) !important;
}

/* ──────────── 14. FOOTER & CHAT INPUT ──────────── */

footer {
    background: transparent !important;
}

footer, footer span, footer a, footer p {
    color: var(--lp-text-3) !important;
}

/* ───── Bottom container ───── */
[data-testid="stBottom"] {
    background: var(--lp-surface) !important;
    border-top: 1px solid var(--lp-border) !important;
    padding: 6px 0 !important;
}

[data-testid="stBottom"] > div,
[data-testid="stBottom"] [data-testid="stVerticalBlock"],
[data-testid="stBottom"] .block-container {
    background: var(--lp-surface) !important;
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    gap: 0 !important;
}

/* ───── Chat Input Container ───── */
[data-testid="stChatInput"] {
    background: var(--lp-input-bg) !important;
    border: 1px solid var(--lp-input-border) !important;
    border-radius: 12px !important;
    max-height: 42px !important;
    min-height: 42px !important;
    padding: 0 !important;
}

/* ───── INPUT TEXT FIX (MAIN SOLUTION) ───── */
[data-testid="stChatInputTextArea"] textarea,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {

    color: var(--lp-text) !important;
    -webkit-text-fill-color: #000000 !important;
    caret-color: var(--lp-text) !important;

    background: transparent !important;

    font-size: 0.875rem !important;
    line-height: 1.4 !important;

    padding: 8px 12px !important;

    /* IMPORTANT FIX: prevent invisibility in dark mode */
    opacity: 1 !important;
}

/* ───── DARK MODE OVERRIDE (SAFE READABILITY) ───── */
[data-theme="dark"] [data-testid="stChatInputTextArea"] textarea,
[data-theme="dark"] [data-testid="stChatInput"] textarea,
[data-theme="dark"] [data-testid="stChatInput"] input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #ffffff !important;
}


/* ───── LIGHT MODE OVERRIDE ───── */
[data-theme="light"] [data-testid="stChatInputTextArea"] textarea,
[data-theme="light"] [data-testid="stChatInput"] textarea,
[data-theme="light"] [data-testid="stChatInput"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    caret-color: #111827 !important;
}

/* ───── Placeholder ───── */
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInput"] input::placeholder {
    color: var(--lp-text-3) !important;
    opacity: 0.75 !important;
}

/* ───── Button ───── */
[data-testid="stChatInput"] button {
    color: var(--lp-text) !important;
    background: transparent !important;
    height: 42px !important;
    width: 42px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;

    padding: 0 !important;
}

/* ───── Send Icon ───── */
[data-testid="stChatInput"] button svg {
    fill: var(--lp-text-2) !important;
}

/* ───── Inner wrapper fix ───── */
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] > div {
    background: var(--lp-input-bg) !important;
    border-color: var(--lp-input-border) !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ───── Bottom text fix ───── */
[data-testid="stBottom"] p,
[data-testid="stBottom"] label {
    color: var(--lp-text) !important;
}
/* ──────────── 16. MODALS / DIALOGS ──────────── */
[data-baseweb="modal"],
[role="dialog"] {
    color: var(--lp-text) !important;
}

[data-baseweb="modal"] > div:first-child,
[data-baseweb="modal-backdrop"] {
    background: rgba(0, 0, 0, 0.7) !important;
    backdrop-filter: blur(6px) !important;
}

[data-baseweb="modal"] > div > div,
[data-baseweb="modal"] [role="document"],
[role="dialog"] > div > div,
[role="dialog"] [role="document"] {
    background: var(--lp-surface) !important;
    color: var(--lp-text) !important;
    border-radius: 16px !important;
    border: 1px solid var(--lp-border) !important;
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5) !important;
}

[data-baseweb="modal"] [data-baseweb="modal-header"],
[data-baseweb="modal"] [data-baseweb="modal-body"],
[data-baseweb="modal"] [data-baseweb="modal-footer"],
[role="dialog"] [data-baseweb="modal-header"],
[role="dialog"] [data-baseweb="modal-body"],
[role="dialog"] [data-baseweb="modal-footer"] {
    background: var(--lp-surface) !important;
    color: var(--lp-text) !important;
}

[data-baseweb="modal"] h1, [data-baseweb="modal"] h2,
[data-baseweb="modal"] h3, [data-baseweb="modal"] h4,
[role="dialog"] h1, [role="dialog"] h2,
[role="dialog"] h3, [role="dialog"] h4 {
    color: var(--lp-text) !important;
}

[data-baseweb="modal"] p, [data-baseweb="modal"] span,
[data-baseweb="modal"] label, [data-baseweb="modal"] li,
[role="dialog"] p, [role="dialog"] span,
[role="dialog"] label, [role="dialog"] li {
    color: var(--lp-text-2) !important;
}

[data-baseweb="modal"] button,
[role="dialog"] button {
    color: var(--lp-text) !important;
    border: 1px solid var(--lp-border) !important;
    background: var(--lp-input-bg) !important;
    border-radius: var(--lp-radius-sm) !important;
    transition: all 0.2s ease !important;
}

[data-baseweb="modal"] button:hover,
[role="dialog"] button:hover {
    background: var(--lp-input-bg) !important;
    filter: brightness(0.95);
    border-color: var(--lp-border) !important;
}

[data-baseweb="modal"] [aria-label="Close"],
[role="dialog"] [aria-label="Close"] {
    color: var(--lp-text-3) !important;
    background: transparent !important;
    border: none !important;
}

[data-baseweb="modal"] [aria-label="Close"]:hover,
[role="dialog"] [aria-label="Close"]:hover {
    color: var(--lp-text) !important;
    background: rgba(148, 163, 184, 0.15) !important;
}

[data-baseweb="modal"] code,
[role="dialog"] code {
    color: var(--lp-accent) !important;
    background: rgba(56, 189, 248, 0.1) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 6px !important;
    padding: 2px 6px !important;
}

/* ──────────── 17. EXPANDERS ──────────── */
[data-testid="stExpander"] {
    background: var(--lp-surface) !important;
    border: 1px solid var(--lp-border) !important;
    border-radius: var(--lp-radius) !important;
    overflow: hidden;
    margin-bottom: var(--lp-spacing-sm) !important;
}

[data-testid="stExpander"] summary {
    padding: 0.75rem var(--lp-spacing-md) !important;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: #000000 !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] summary svg {
    fill: var(--lp-text-3) !important;
}

[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p,
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] li {
    color: #000000 !important;
}

/* ──────────── 18. CHAT MESSAGES ──────────── */
[data-testid="stChatMessage"] {
    background: var(--lp-surface) !important;
    border: 1px solid var(--lp-border) !important;
    border-radius: var(--lp-radius) !important;
    padding: var(--lp-spacing-md) !important;
    margin-bottom: var(--lp-spacing-sm) !important;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span {
    color: var(--lp-text) !important;
}

/* ──────────── 19. ALERTS ──────────── */
[data-testid="stAlert"] {
    border-radius: var(--lp-radius) !important;
    border: none !important;
}

/* ──────────── 20. HORIZONTAL RULES ──────────── */
hr {
    border: 0;
    border-top: 1px solid var(--lp-hr);
    margin: var(--lp-spacing-lg) 0;
}

[data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] {
    gap: var(--lp-spacing-sm);
}

/* ──────────── 21. SMOOTH TRANSITIONS ──────────── */
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
[data-testid="stHeader"],
[data-testid="stBottom"],
.metric-card,
.kpi-card,
[data-testid="stExpander"],
[data-testid="stChatMessage"],
[data-testid="stPlotlyChart"] {
    transition: background 0.3s ease, color 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
}

/* ──────────── 22. RESPONSIVE BREAKPOINTS ──────────── */
@media (max-width: 992px) {
    .main .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }

    .metric-card {
        min-height: 110px;
    }

    .hero-banner {
        padding: 1.6rem 1.4rem;
    }

    .hero-banner .hero-title {
        font-size: 1.5rem;
    }

    .kpi-row {
        gap: 10px;
    }
}

@media (max-width: 640px) {
    .main .block-container {
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }

    h1 {
        font-size: 1.4rem !important;
    }

    h2 {
        font-size: 1.15rem !important;
    }

    h3 {
        font-size: 1rem !important;
    }

    .hero-banner {
        padding: 1.2rem 1rem;
        border-radius: 14px;
    }

    .hero-banner .hero-title {
        font-size: 1.3rem;
    }

    .section-header {
        gap: 6px;
    }

    .section-header .section-icon {
        font-size: 1.1rem;
    }

    .section-header .section-title {
        font-size: 1rem;
    }

    .kpi-row {
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    }
}

/* ──────────── 23. SUBHEADER ──────────── */
[data-testid="stSubheader"] {
    padding-top: 0.8rem !important;
}
/* Yeh rule direct toggle status se selectbox handle karega */
[data-testid="stSidebar"] div[data-baseweb="select"] div {
    color: var(--lp-input-text) !important;
    -webkit-text-fill-color: var(--lp-input-text) !important;
}
</style>
""", unsafe_allow_html=True)


if not st.session_state.get("dark_mode", True):
    st.markdown("""<style>
    :root {
        --lp-bg: linear-gradient(180deg, #f0f4f8 0%, #e8edf3 100%);
        --lp-sidebar: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
        --lp-surface: #ffffff;
        --lp-surface-grad: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        --lp-text: #0f172a;
        --lp-text-2: #475569;
        --lp-text-3: #64748b;
        --lp-border: #e2e8f0;
        --lp-input-bg: #ffffff;
        --lp-input-text: #0f172a;
        --lp-input-border: #cbd5e1;
        --lp-header-bg: rgba(248, 250, 252, 0.9);
        --lp-card-shadow: 0 2px 8px rgba(15, 23, 42, 0.07);
        --lp-chart-border: #e2e8f0;
        --lp-hr: #e2e8f0;
        --lp-panel-bg: linear-gradient(135deg, #f1f5f9 0%, #e8edf3 100%);
        --lp-panel-border: rgba(47, 124, 255, 0.12);
        --lp-panel-shadow: 0 2px 12px rgba(15, 23, 42, 0.06);
        --lp-panel-cell: rgba(15, 23, 42, 0.04);
        --lp-rec-bg: linear-gradient(135deg, #e0ecf8 0%, #dbe4f0 100%);
        --lp-accent: #0284c7;
    }
    .metric-card:hover,
    .kpi-card:hover {
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
    }
    .section-header .section-badge {
        color: #2f7cff;
    }
    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
    }
    [data-testid="stBottom"] {
        background: #f8fafc !important;
        border-top: 1px solid #e2e8f0 !important;
        padding: 6px 0 !important;
    }
    [data-testid="stBottom"] > div,
    [data-testid="stBottom"] [data-testid="stVerticalBlock"] {
        background: #f8fafc !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        gap: 0 !important;
    }
    .stButton > button {
        box-shadow: 0 8px 20px rgba(0, 111, 255, 0.2) !important;
    }
    .stButton > button:hover {
        box-shadow: 0 12px 28px rgba(0, 111, 255, 0.3) !important;
    }
    .hero-banner {
        background: linear-gradient(135deg, #e8edf3 0%, #dce4f0 50%, #eaeff5 100%) !important;
        border-color: rgba(47, 124, 255, 0.12) !important;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.06) !important;
    }
    .hero-banner::before {
        background: radial-gradient(ellipse, rgba(47, 124, 255, 0.04) 0%, transparent 70%) !important;
    }
    .hero-banner .hero-badge {
        background: rgba(47, 124, 255, 0.08) !important;
        border-color: rgba(47, 124, 255, 0.15) !important;
        color: #2f7cff !important;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] button span,
    [data-testid="stExpandSidebarButton"],
    [data-testid="stExpandSidebarButton"] span {
        color: #334155 !important;
        opacity: 1 !important; /* Always visible */
    }
    [data-testid="stSidebarCollapseButton"] button:hover,
    [data-testid="stSidebarCollapseButton"] button:active,
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stExpandSidebarButton"]:active {
        color: #0f172a !important;
        opacity: 0.9 !important;
    }
    .hero-banner .hero-title { color: #0f172a !important; }
    .hero-banner .hero-subtitle { color: #475569 !important; }
    .hero-banner .hero-desc { color: #64748b !important; }
    .hero-banner .hero-desc strong { color: #2f7cff !important; }
    </style>""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (cached — deserialized once per process, reused on every rerun)
# =========================
try:
    artifacts = load_artifacts_cached()
    model = artifacts["model"]
    features = artifacts["feature_names"]
    encoders = artifacts["encoders"]
    _best_model_display_name = artifacts.get("best_model_name", "ML Model")

    validate_feature_alignment(features)

    st.session_state.model_name = _best_model_display_name

    log.info("Model loaded: %s (%d features)", _best_model_display_name, len(features))
except FileNotFoundError as e:
    st.error(f"Model not found. Please run `python train.py` first.\n\n{e}")
    st.stop()
except ValueError as e:
    st.error(f"Model artifact is invalid.\n\n{e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# =========================
# HERO SECTION
# =========================
st.markdown(
    f"""<div class="hero-banner">
        <div class="hero-badge">AI-Powered Risk Analytics</div>
        <div class="hero-title">🏦 AI Loan Intelligence System</div>
        <div class="hero-subtitle">Advanced Banking Risk Prediction Engine</div>
        <div class="hero-desc">
            Production-grade ML pipeline with multi-model evaluation, real-time risk scoring,
            and comprehensive financial analytics. Powered by <strong style="color:#60a5fa;">{_best_model_display_name}</strong>.
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# =========================
# KPI SUMMARY CARDS
# =========================
_kpi_csv = str(MODEL_COMPARISON_CSV)
_kpi_accuracy = "—"
_kpi_precision_avg = "—"
_kpi_model_count = "—"
_kpi_best_name = _best_model_display_name

if os.path.exists(_kpi_csv):
    try:
        _kpi_df = pd.read_csv(_kpi_csv)
        _kpi_model_count = str(len(_kpi_df))
        if "Accuracy" in _kpi_df.columns:
            _kpi_accuracy = f"{_kpi_df['Accuracy'].max():.2%}"
        if "Precision" in _kpi_df.columns:
            _kpi_precision_avg = f"{_kpi_df['Precision'].mean():.2%}"
    except Exception as e:
        log.warning("Failed to load KPI data from %s: %s", _kpi_csv, e)

_kpi_dataset_size = "—"
if DATASET_PATH.exists():
    try:
        _kpi_dataset_size = f"{len(pd.read_csv(str(DATASET_PATH), usecols=[0])):,}"
    except Exception as e:
        log.warning("Failed to read dataset size: %s", e)

st.markdown(
    f"""<div class="kpi-row">
        <div class="kpi-card kpi-blue">
            <div class="kpi-label">Best Model</div>
            <div class="kpi-value">{_kpi_best_name}</div>
            <div class="kpi-sub">Top performer by ROC AUC</div>
        </div>
        <div class="kpi-card kpi-green">
            <div class="kpi-label">Best Accuracy</div>
            <div class="kpi-value">{_kpi_accuracy}</div>
            <div class="kpi-sub">Highest across all models</div>
        </div>
        <div class="kpi-card kpi-purple">
            <div class="kpi-label">Avg Precision</div>
            <div class="kpi-value">{_kpi_precision_avg}</div>
            <div class="kpi-sub">Mean of all models</div>
        </div>
        <div class="kpi-card kpi-cyan">
            <div class="kpi-label">Dataset Size</div>
            <div class="kpi-value">{_kpi_dataset_size}</div>
            <div class="kpi-sub">Training records</div>
        </div>
        <div class="kpi-card kpi-amber">
            <div class="kpi-label">Predictions</div>
            <div class="kpi-value">{st.session_state.prediction_counter}</div>
            <div class="kpi-sub">Session predictions</div>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# =========================
# SIDEBAR — LOAN APPLICATION FORM
# =========================
st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="sidebar-form-title">'
    '<span class="bank-icon">🏦</span>'
    '<p class="form-heading">LOAN APPLICATION FORM</p>'
    '<p class="bank-name">Global AI FinTech Bank</p>'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    f'<div class="sidebar-app-id">'
    f'<div class="app-id-label">Application Reference</div>'
    f'<div class="app-id-value">{st.session_state.application_id}</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ━━━━━━ PERSONAL INFORMATION ━━━━━━
st.sidebar.markdown(
    '<div class="sidebar-section-header">👤 Personal Information</div>',
    unsafe_allow_html=True,
)

full_name = st.sidebar.text_input("Full Name", placeholder="e.g. Ahmed Khan")
cnic_number = st.sidebar.text_input("CNIC / National ID", placeholder="e.g. 12345-6789012-3")
phone_number = st.sidebar.text_input("Phone Number", placeholder="e.g. +92 300 1234567")
email_address = st.sidebar.text_input("Email Address", placeholder="e.g. name@example.com")

_name_invalid = bool(full_name) and len(full_name.strip()) < 2
_cnic_invalid = bool(cnic_number) and not re.match(r'^\d{5}-?\d{7}-?\d{1}$', cnic_number.strip())
_phone_invalid = bool(phone_number) and not re.match(r'^[\+]?[\d\s\-]{7,15}$', phone_number.strip())
_email_invalid = bool(email_address) and not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email_address.strip())

if _name_invalid:
    st.sidebar.markdown(
        '<div class="validation-warn">⚠ Name must be at least 2 characters</div>',
        unsafe_allow_html=True,
    )

if _cnic_invalid:
    st.sidebar.markdown(
        '<div class="validation-warn">⚠ Enter a valid CNIC (e.g. 12345-6789012-3)</div>',
        unsafe_allow_html=True,
    )

if _phone_invalid:
    st.sidebar.markdown(
        '<div class="validation-warn">⚠ Enter a valid phone number</div>',
        unsafe_allow_html=True,
    )

if _email_invalid:
    st.sidebar.markdown(
        '<div class="validation-warn">⚠ Enter a valid email address</div>',
        unsafe_allow_html=True,
    )

age = st.sidebar.number_input("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", encoders['gender'].classes_)
marital_status = st.sidebar.selectbox("Marital Status", encoders['marital_status'].classes_)

# ━━━━━━ EMPLOYMENT & EDUCATION ━━━━━━
st.sidebar.markdown(
    '<div class="sidebar-section-header">💼 Employment & Education</div>',
    unsafe_allow_html=True,
)

education_level = st.sidebar.selectbox("Education Level", encoders['education_level'].classes_)
employment_status = st.sidebar.selectbox("Employment Status", encoders['employment_status'].classes_)

# ━━━━━━ FINANCIAL INFORMATION ━━━━━━
st.sidebar.markdown(
    '<div class="sidebar-section-header">💰 Financial Information</div>',
    unsafe_allow_html=True,
)

annual_income = st.sidebar.number_input("Annual Income ($)", 0, 1000000, 50000)
monthly_income = st.sidebar.number_input("Monthly Income ($)", 0, 100000, 5000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
debt_ratio = st.sidebar.slider("Debt to Income Ratio", 0.0, 1.0, 0.3)

# ━━━━━━ LOAN DETAILS ━━━━━━
st.sidebar.markdown(
    '<div class="sidebar-section-header">🏦 Loan Details</div>',
    unsafe_allow_html=True,
)

loan_amount = st.sidebar.number_input("Loan Amount ($)", 100, 1000000, 10000)
loan_purpose = st.sidebar.selectbox("Loan Purpose", encoders['loan_purpose'].classes_)
grade_subgrade = st.sidebar.selectbox("Grade / Subgrade", encoders['grade_subgrade'].classes_)
interest_rate = st.sidebar.slider("Interest Rate (%)", 1.0, 25.0, 10.0)
loan_term = st.sidebar.selectbox("Loan Term", [36, 60], format_func=lambda x: f"{x} Months")
installment = st.sidebar.number_input("Monthly Installment ($)", 1.0, 100000.0, 450.0, step=10.0)

# ━━━━━━ CREDIT & ACCOUNT INFO ━━━━━━
st.sidebar.markdown(
    '<div class="sidebar-section-header">📋 Credit & Account Info</div>',
    unsafe_allow_html=True,
)

num_of_open_accounts = st.sidebar.number_input("Open Accounts", 0, 50, 5)
total_credit_limit = st.sidebar.number_input("Total Credit Limit ($)", 0.0, 10000000.0, 40000.0, step=1000.0)
current_balance = st.sidebar.number_input("Current Balance ($)", 0.0, 10000000.0, 18000.0, step=500.0)
delinquency_history = st.sidebar.number_input("Delinquency History", 0, 20, 0)
public_records = st.sidebar.number_input("Public Records", 0, 10, 0)
num_of_delinquencies = st.sidebar.number_input("Number of Delinquencies", 0, 20, 0)

st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

# =========================
# BUILD RAW INPUT DICT
# =========================
raw_input = {
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education_level": education_level,
    "employment_status": employment_status,
    "annual_income": annual_income,
    "monthly_income": monthly_income,
    "credit_score": credit_score,
    "debt_to_income_ratio": debt_ratio,
    "loan_amount": loan_amount,
    "loan_purpose": loan_purpose,
    "grade_subgrade": grade_subgrade,
    "interest_rate": interest_rate,
    "loan_term": loan_term,
    "installment": installment,
    "num_of_open_accounts": num_of_open_accounts,
    "total_credit_limit": total_credit_limit,
    "current_balance": current_balance,
    "delinquency_history": delinquency_history,
    "public_records": public_records,
    "num_of_delinquencies": num_of_delinquencies,
}

# =========================
# STALE-PREDICTION DETECTION
# =========================
_current_input_hash = _hash_inputs(raw_input)
_prediction_is_stale = (
    st.session_state.raw_pred is not None
    and st.session_state.prediction_input_hash != _current_input_hash
)

# =========================
# PRE-PREDICTION VALIDATION (reuses sidebar validation flags)
# =========================
_personal_warnings = []
if not full_name or len(full_name.strip()) < 2:
    _personal_warnings.append("Full Name is required (at least 2 characters).")
if _cnic_invalid:
    _personal_warnings.append("CNIC format is invalid.")
if _phone_invalid:
    _personal_warnings.append("Phone number format is invalid.")
if _email_invalid:
    _personal_warnings.append("Email address format is invalid.")

# =========================
# PREDICTION (only on button click — never on rerun)
# =========================
if st.button("🚀 Predict Loan Status"):
    if _personal_warnings:
        for _pw in _personal_warnings:
            st.warning(_pw)

    try:
        result = run_prediction(raw_input, model, encoders, features)

        # ── Probability sanity check ──
        if not (0.0 <= result.probability <= 1.0):
            log.error("Probability out of range: %.6f", result.probability)
            st.error("Model returned an invalid probability. Please contact support.")
        elif result.raw_pred not in (0, 1):
            log.error("Invalid raw_pred value: %s", result.raw_pred)
            st.error("Model returned an invalid prediction. Please contact support.")
        else:
            # ── Atomic state update: all prediction outputs written together ──
            st.session_state.raw_pred = result.raw_pred
            st.session_state.raw_proba = result.probability
            st.session_state.risk_level = result.risk_level
            st.session_state.risk_color = result.risk_color
            st.session_state.risk_emoji = result.risk_emoji
            st.session_state.confidence = result.confidence
            st.session_state.prediction_counter += 1
            st.session_state.prediction_input_hash = _current_input_hash

            st.session_state.prediction_timestamp = datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # ── Snapshot applicant data at prediction time (not at display time) ──
            _prediction_text = "Approved" if result.raw_pred == 1 else "Rejected"
            st.session_state.last_prediction = _prediction_text
            st.session_state.last_probability = result.probability
            st.session_state.last_applicant_data = {
                "Application ID": st.session_state.application_id,
                "Full Name": full_name,
                "CNIC": cnic_number,
                "Phone": phone_number,
                "Email": email_address,
                "Age": age,
                "Gender": gender,
                "Marital Status": marital_status,
                "Education Level": education_level,
                "Employment Status": employment_status,
                "Annual Income": annual_income,
                "Monthly Income": monthly_income,
                "Loan Amount": loan_amount,
                "Credit Score": credit_score,
                "Debt To Income Ratio": debt_ratio,
                "Interest Rate": interest_rate,
                "Loan Purpose": loan_purpose,
                "Grade / Subgrade": grade_subgrade,
                "Loan Term": f"{loan_term} Months",
                "Monthly Installment": installment,
                "Open Accounts": num_of_open_accounts,
                "Total Credit Limit": total_credit_limit,
                "Current Balance": current_balance,
                "Delinquency History": delinquency_history,
                "Public Records": public_records,
                "Delinquencies": num_of_delinquencies,
            }

            # ── Invalidate stale PDF from previous prediction ──
            st.session_state.pdf_ready = False
            st.session_state.pdf_bytes = None

            # ── Clear stale chat history so assistant doesn't reference previous applicant ──
            st.session_state.chat_history = []

            _prediction_is_stale = False

    except ValueError as ve:
        log.warning("Input validation failed: %s", ve)
        st.error(str(ve))
    except RuntimeError as re_err:
        log.error("Prediction engine error: %s", re_err)
        st.error(f"Prediction engine error: {re_err}")
    except Exception as exc:
        log.error("Unexpected prediction error: %s", exc, exc_info=True)
        st.error(f"Unexpected error: {exc}")

# =========================
# RESULTS DISPLAY (reads entirely from session_state)
# =========================
if st.session_state.raw_pred is not None:
    pred = st.session_state.raw_pred
    proba = st.session_state.raw_proba

    # ── Stale-prediction warning ──
    if _prediction_is_stale:
        st.warning(
            "⚠ You have changed inputs since the last prediction. "
            "Click **Predict Loan Status** to update results."
        )

    st.markdown("---")

    # =========================
    # METRICS CARDS
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    _risk_level = st.session_state.risk_level or "N/A"
    _risk_emoji = st.session_state.risk_emoji or ""
    _confidence = st.session_state.confidence or "N/A"
    status = "APPROVED ✅" if pred == 1 else "REJECTED ❌"

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Approval Probability</div>
                <div class="metric-value">{proba:.2%}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Risk Level</div>
                <div class="metric-value">{_risk_emoji} {_risk_level}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Application Status</div>
                <div class="metric-value">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{_confidence}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================
    # GAUGE CHART (RISK METER)
    # =========================
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><span class="section-title">Risk Meter</span></div>', unsafe_allow_html=True)

    _ct = _chart_theme()

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=proba * 100,
        number={'suffix': '%', 'font': {'size': 42, 'color': _ct["title"], 'family': 'Inter, sans-serif'}},
        title={'text': "Loan Approval Probability", 'font': {'size': 15, 'color': _ct["axis"]}},
        delta={'reference': 50, 'increasing': {'color': '#10b981'}, 'decreasing': {'color': '#ef4444'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': _ct["muted"],
                     'tickfont': {'color': _ct["axis"], 'size': 11}},
            'bar': {'color': '#2f7cff', 'thickness': 0.75},
            'bgcolor': _ct["surface"],
            'borderwidth': 0,
            'steps': [
                {'range': [0, 33], 'color': '#fecaca'},
                {'range': [33, 66], 'color': '#fef3c7'},
                {'range': [66, 100], 'color': '#bbf7d0'}
            ],
            'threshold': {
                'line': {'color': _ct["title"], 'width': 3},
                'thickness': 0.8,
                'value': proba * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor=_ct["bg"],
        plot_bgcolor=_ct["bg"],
        font=dict(color=_ct["text"], family="Inter, sans-serif"),
        height=400,
        margin=dict(l=30, r=30, t=60, b=20),
    )

    st.plotly_chart(fig, width="stretch")

    # =========================
    # FEATURE IMPORTANCE (COLORFUL)
    # =========================
    st.markdown('<div class="section-header"><span class="section-icon">📌</span><span class="section-title">AI Feature Importance</span></div>', unsafe_allow_html=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]

        df_imp = pd.DataFrame({
            "Feature": [features[i] for i in indices],
            "Importance": importances[indices]
        })

        fintech_colors = ['#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1', '#8b5cf6',
                           '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#ef4444']

        fig2 = go.Figure(go.Bar(
            x=df_imp["Importance"],
            y=df_imp["Feature"],
            orientation='h',
            marker=dict(
                color=fintech_colors[:len(df_imp)],
                line=dict(width=0),
                cornerradius=4,
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))

        fig2.update_layout(
            paper_bgcolor=_ct["bg"],
            plot_bgcolor=_ct["bg"],
            font=dict(color=_ct["text"], family="Inter, sans-serif"),
            height=420,
            margin=dict(l=20, r=20, t=10, b=20),
            yaxis=dict(
                tickfont=dict(size=12, color=_ct["text"]),
                gridcolor=_ct["grid"],
            ),
            xaxis=dict(
                title=dict(text="Importance Score", font=dict(size=13, color=_ct["axis"])),
                tickfont=dict(color=_ct["axis"]),
                gridcolor=_ct["grid"],
                showgrid=True,
            ),
            bargap=0.25,
        )

        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("Feature importance is not available for this model type.")

    # =========================
    # AI EXPLANATION
    # =========================
    st.markdown('<div class="section-header"><span class="section-icon">🧠</span><span class="section-title">AI Decision Insight</span></div>', unsafe_allow_html=True)

    if pred == 1:
        st.success("✔ Strong financial profile detected. Low default probability.")
    else:
        st.error("⚠ High risk detected. Applicant may default based on ML patterns.")

    # =========================
    # VARIABLES FOR PDF AND CHATBOT (read from session_state snapshot)
    # =========================
    applicant_data = st.session_state.last_applicant_data
    prediction_text = st.session_state.last_prediction
    risk_level = _risk_level
    probability = proba

    # =========================
    # PDF REPORT DOWNLOAD (gated behind fresh prediction)
    # =========================
    st.markdown("---")
    st.markdown('<div class="section-header"><span class="section-icon">📄</span><span class="section-title">AI Loan Assessment Report</span></div>', unsafe_allow_html=True)

    if _prediction_is_stale:
        st.info("Run a new prediction to enable report generation.")
    elif not applicant_data or prediction_text is None or probability is None:
        st.info("Complete a prediction to enable report generation.")
    else:
        pdf_download_name = _build_pdf_filename(
            st.session_state.application_id,
            applicant_data.get("Full Name", "") if applicant_data else "",
        )

        if st.button("📥 Generate & Download AI Report"):
            try:
                model_name = st.session_state.model_name
                pdf_path = generate_pdf_report(
                    applicant_data,
                    prediction_text,
                    probability,
                    risk_level,
                    model_name,
                    output_path="outputs/loan_report.pdf"
                )

                with open(pdf_path, "rb") as pdf_file:
                    st.session_state.pdf_bytes = pdf_file.read()

                st.session_state.pdf_ready = True
                st.session_state.pdf_file_name = pdf_download_name
            except FileNotFoundError as e:
                log.error("PDF report file not found: %s", e)
                st.error("Report generation failed. Output directory may be missing.")
            except Exception as e:
                log.error("PDF generation failed: %s", e, exc_info=True)
                st.error("Unable to generate report. Please try again.")

        if st.session_state.get("pdf_ready") and st.session_state.pdf_bytes:
            st.success("✅ Professional AI Report Generated Successfully.")
            pdf_b64 = base64.b64encode(st.session_state.pdf_bytes).decode()
            dl_name = st.session_state.pdf_file_name
            st.markdown(
                f'<a href="data:application/pdf;base64,{pdf_b64}" '
                f'download="{dl_name}" '
                f'style="display:inline-flex;align-items:center;justify-content:center;'
                f'gap:0.5rem;padding:0.6rem 1.2rem;background-color:#2f7cff;color:#ffffff;'
                f'border:none;border-radius:0.5rem;text-decoration:none;font-weight:600;'
                f'font-size:0.875rem;width:100%;cursor:pointer;transition:background 0.2s;"'
                f' onmouseover="this.style.backgroundColor=\'#1a5ec7\'"'
                f' onmouseout="this.style.backgroundColor=\'#2f7cff\'">'
                f'⬇️ Download PDF Report</a>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption("💡 AI Powered FinTech System | Advanced Machine Learning Project")



# =========================
# MODEL COMPARISON DASHBOARD
# =========================
st.markdown("---")
st.markdown(
    '<div class="section-header">'
    '<span class="section-icon">\U0001f3c6</span>'
    '<span class="section-title">Machine Learning Model Comparison</span>'
    '<span class="section-badge">Dashboard</span></div>',
    unsafe_allow_html=True,
)

_csv_path = str(MODEL_COMPARISON_CSV)
_dash_pkl_path = str(DASHBOARD_DATA_PATH)

_mc = None
if os.path.exists(_csv_path):
    try:
        _mc = pd.read_csv(_csv_path)
    except Exception as e:
        log.error("Failed to load model comparison CSV: %s", e)

if _mc is not None and not _mc.empty:

    _col_renames = {"ROC-AUC": "ROC AUC", "F1": "F1 Score"}
    _mc.rename(columns={k: v for k, v in _col_renames.items() if k in _mc.columns}, inplace=True)

    if "Model" not in _mc.columns:
        if _mc.index.dtype == object and _mc.index.name != "Model":
            _mc = _mc.reset_index()
            _mc.rename(columns={_mc.columns[0]: "Model"}, inplace=True)

    _metric_cols = [c for c in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"] if c in _mc.columns]
    for _c in _metric_cols:
        _mc[_c] = _mc[_c].astype(float).round(4)

    _sort_key = "Accuracy" if "Accuracy" in _mc.columns else (_metric_cols[0] if _metric_cols else _mc.columns[0])
    _mc = _mc.sort_values(
        _sort_key,
        ascending=False,
    ).reset_index(drop=True)

    _best_idx = 0
    if "ROC AUC" in _mc.columns:
        _best_idx = _mc["ROC AUC"].idxmax()
    elif "Accuracy" in _mc.columns:
        _best_idx = _mc["Accuracy"].idxmax()
    _best = _mc.loc[_best_idx]

    _palette = {
        "Accuracy": "#3b82f6",
        "Precision": "#8b5cf6",
        "Recall": "#06b6d4",
        "F1 Score": "#f59e0b",
        "ROC AUC": "#10b981",
    }
    _model_palette = CB_PALETTE
    _ct = _chart_theme()

    _dash_data = None
    if os.path.exists(_dash_pkl_path):
        try:
            with open(_dash_pkl_path, "rb") as _f:
                _dash_data = pickle.load(_f)
        except Exception as e:
            log.warning("Failed to load dashboard data: %s", e)

    # ── SIDEBAR FILTERS ──
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<p style="color:var(--lp-text);font-weight:700;font-size:0.85rem;'
        'text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">'
        '\U0001f4ca Dashboard Filters</p>',
        unsafe_allow_html=True,
    )
    _sel_metric = st.sidebar.selectbox(
        "Highlight Metric",
        _metric_cols,
        index=_metric_cols.index("ROC AUC") if "ROC AUC" in _metric_cols else 0,
        key="dash_metric",
    )
    _view_mode = st.sidebar.radio(
        "View Mode",
        ["All Models", "Top 2 Models", "Best Model Only"],
        index=0,
        key="dash_view",
    )
    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if _view_mode == "Best Model Only":
        _df = _mc.loc[[_best_idx]].reset_index(drop=True)
    elif _view_mode == "Top 2 Models":
        _sort_col = _sel_metric if _sel_metric in _mc.columns else _metric_cols[0]
        _df = _mc.nlargest(2, _sort_col).reset_index(drop=True)
    else:
        _df = _mc.copy()

    _orig_best_model = _best["Model"]

    # ────────────────────────────────────────────
    # PART 1 — PERFORMANCE METRICS TABLE
    # ────────────────────────────────────────────
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4ca</span>'
        '<span class="section-title">Performance Metrics Table</span>'
        '<span class="section-badge">Table</span></div>',
        unsafe_allow_html=True,
    )

    _display = _df.copy()

    def _highlight_best(row):
        if row.get("Model") == _orig_best_model:
            return ["background-color:rgba(47,124,255,0.12);font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        _display.style.apply(_highlight_best, axis=1).format(
            {c: "{:.2f}" for c in _metric_cols if c in _display.columns}
        ),
        width="stretch",
        hide_index=True,
    )

    # ────────────────────────────────────────────
    # PART 2 — BEST MODEL SUMMARY CARD
    # ────────────────────────────────────────────
    st.markdown("---")

    def _rank_badge(rank):
        if rank == 1:
            return '<span style="background:#f59e0b;color:#fff;font-size:0.6rem;padding:2px 6px;border-radius:8px;font-weight:700;">1st</span>'
        if rank == 2:
            return '<span style="background:#94a3b8;color:#fff;font-size:0.6rem;padding:2px 6px;border-radius:8px;font-weight:700;">2nd</span>'
        if rank == 3:
            return '<span style="background:#b45309;color:#fff;font-size:0.6rem;padding:2px 6px;border-radius:8px;font-weight:700;">3rd</span>'
        return ""

    _metric_cells = ""
    for _m in _metric_cols:
        _rank = int((_mc[_m].rank(ascending=False)).loc[_best_idx]) if _m in _mc.columns else 0
        _badge = _rank_badge(_rank)
        _metric_cells += (
            f'<div style="background:var(--lp-panel-cell);border-radius:10px;'
            f'padding:14px;text-align:center;">'
            f'<div style="color:var(--lp-text-3);font-size:0.68rem;text-transform:uppercase;'
            f'letter-spacing:1px;margin-bottom:4px;">{_m}</div>'
            f'<div style="color:{_palette.get(_m, "#fff")};font-size:1.3rem;font-weight:700;">'
            f'{_best.get(_m, 0):.2f}</div>'
            f'<div style="margin-top:4px;">{_badge}</div></div>'
        )

    st.markdown(
        f"""<div style="background:var(--lp-panel-bg);
        border-radius:16px;padding:28px 32px;margin:18px 0 24px 0;
        border:1px solid var(--lp-panel-border);
        box-shadow:var(--lp-panel-shadow);">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:18px;flex-wrap:wrap;">
            <span style="font-size:2rem;">\U0001f3c6</span>
            <div>
                <div style="color:var(--lp-text-3);font-size:0.78rem;text-transform:uppercase;
                letter-spacing:1.5px;font-weight:600;">Best Performing Model</div>
                <div style="color:var(--lp-text);font-size:1.5rem;font-weight:700;
                margin-top:2px;">{_best['Model']}</div>
            </div>
            <span style="margin-left:auto;display:inline-flex;align-items:center;gap:6px;
            background:#059669;color:#fff;padding:5px 14px;border-radius:20px;
            font-size:0.75rem;font-weight:600;letter-spacing:0.5px;">
            <span style="width:8px;height:8px;background:#34d399;border-radius:50%;
            display:inline-block;"></span>
            PRODUCTION READY</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;">
            {_metric_cells}
        </div></div>""",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────
    # PART 3 — INDIVIDUAL METRIC BAR CHARTS
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4ca</span>'
        '<span class="section-title">Individual Metric Performance</span>'
        '<span class="section-badge">Per-Metric</span></div>',
        unsafe_allow_html=True,
    )

    def _make_metric_bar(df, metric, color, ct):
        _sorted = df.sort_values(metric, ascending=True)
        _r, _g, _b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        _bar_colors = [
            color if name == _orig_best_model else f"rgba({_r},{_g},{_b},0.55)"
            for name in _sorted["Model"]
        ]
        fig = go.Figure(go.Bar(
            y=_sorted["Model"],
            x=_sorted[metric],
            orientation="h",
            marker=dict(color=_bar_colors, cornerradius=4),
            text=[f"{v:.2f}" for v in _sorted[metric]],
            textposition="outside",
            textfont=dict(size=11, color=ct["text"]),
            hovertemplate="<b>%{y}</b><br>" + metric + ": %{x:.4f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(
                text=metric,
                font=dict(size=14, color=color, family="Inter,sans-serif"),
            ),
            paper_bgcolor=ct["bg"],
            plot_bgcolor=ct["bg"],
            xaxis=dict(
                range=[0, max(df[metric].max() * 1.15, 0.01)],
                showgrid=True,
                gridcolor=ct["grid"],
                tickfont=dict(size=10, color=ct["axis"]),
            ),
            yaxis=dict(tickfont=dict(size=11, color=ct["text"], family="Inter,sans-serif")),
            margin=dict(l=10, r=40, t=36, b=10),
            height=max(180, 50 * len(df) + 60),
            bargap=0.3,
        )
        return fig

    _row1_c1, _row1_c2 = st.columns(2)
    with _row1_c1:
        if "Accuracy" in _mc.columns:
            st.plotly_chart(_make_metric_bar(_mc, "Accuracy", _palette["Accuracy"], _ct), width="stretch")
    with _row1_c2:
        if "Precision" in _mc.columns:
            st.plotly_chart(_make_metric_bar(_mc, "Precision", _palette["Precision"], _ct), width="stretch")

    _row2_c1, _row2_c2 = st.columns(2)
    with _row2_c1:
        if "Recall" in _mc.columns:
            st.plotly_chart(_make_metric_bar(_mc, "Recall", _palette["Recall"], _ct), width="stretch")
    with _row2_c2:
        if "F1 Score" in _mc.columns:
            st.plotly_chart(_make_metric_bar(_mc, "F1 Score", _palette["F1 Score"], _ct), width="stretch")

    if "ROC AUC" in _mc.columns:
        st.plotly_chart(_make_metric_bar(_mc, "ROC AUC", _palette["ROC AUC"], _ct), width="stretch")

    # ────────────────────────────────────────────
    # PART 4 — GROUPED COMPARISON CHART
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4c8</span>'
        '<span class="section-title">Model Metrics Comparison</span>'
        '<span class="section-badge">Grouped</span></div>',
        unsafe_allow_html=True,
    )

    _grouped_metrics = [m for m in ["Accuracy", "Precision", "Recall", "F1 Score"] if m in _df.columns]
    if _grouped_metrics:
        _fig_g = go.Figure()
        for _i, _gm in enumerate(_grouped_metrics):
            _fig_g.add_trace(go.Bar(
                name=_gm,
                x=_df["Model"], y=_df[_gm],
                marker=dict(color=_palette.get(_gm, CB_PALETTE[_i]), cornerradius=5),
                text=[f"{v:.2f}" for v in _df[_gm]],
                textposition="outside",
                textfont=dict(size=11, color=_ct["text"]),
                hovertemplate="<b>%{x}</b><br>" + _gm + ": %{y:.4f}<extra></extra>",
            ))
        _fig_g.update_layout(
            barmode="group",
            title=dict(
                text="Accuracy / Precision / Recall / F1",
                font=dict(size=16, color=_ct["title"], family="Inter,sans-serif"),
            ),
            plot_bgcolor=_ct["bg"], paper_bgcolor=_ct["bg"],
            yaxis=dict(
                range=[0, 1.15], showgrid=True, gridcolor=_ct["grid"],
                title=dict(text="Score", font=dict(color=_ct["axis"])),
                tickfont=dict(size=11, color=_ct["axis"]),
            ),
            xaxis=dict(tickfont=dict(size=12, color=_ct["text"], family="Inter,sans-serif")),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                font=dict(size=12, color=_ct["legend"]),
            ),
            margin=dict(l=10, r=10, t=60, b=10), height=420,
        )
        st.plotly_chart(_fig_g, width="stretch")

    # ────────────────────────────────────────────
    # PART 5 — ROC CURVE COMPARISON
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4c9</span>'
        '<span class="section-title">ROC Curve Comparison</span>'
        '<span class="section-badge">AUC</span></div>',
        unsafe_allow_html=True,
    )

    if _dash_data and "roc_curves" in _dash_data and _dash_data["roc_curves"]:
        _fig_roc = go.Figure()
        for _ri, (_roc_name, _roc_vals) in enumerate(_dash_data["roc_curves"].items()):
            _mclr = _model_palette[_ri % len(_model_palette)]
            _fig_roc.add_trace(go.Scatter(
                x=_roc_vals["fpr"], y=_roc_vals["tpr"],
                mode="lines",
                name=f'{_roc_name}  (AUC = {_roc_vals["auc"]:.4f})',
                line=dict(color=_mclr, width=2.5),
                hovertemplate=f"<b>{_roc_name}</b><br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
            ))
        _fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=_ct["muted"], width=1.5, dash="dash"),
            name="Random Baseline", showlegend=True,
            hoverinfo="skip",
        ))
        _fig_roc.update_layout(
            title=dict(
                text="Receiver Operating Characteristic (ROC) Curves",
                font=dict(size=16, color=_ct["title"], family="Inter,sans-serif"),
            ),
            xaxis=dict(
                title=dict(text="False Positive Rate", font=dict(color=_ct["axis"])),
                showgrid=True, gridcolor=_ct["grid"],
                tickfont=dict(size=11, color=_ct["axis"]), range=[0, 1],
            ),
            yaxis=dict(
                title=dict(text="True Positive Rate", font=dict(color=_ct["axis"])),
                showgrid=True, gridcolor=_ct["grid"],
                tickfont=dict(size=11, color=_ct["axis"]), range=[0, 1.02],
            ),
            plot_bgcolor=_ct["bg"], paper_bgcolor=_ct["bg"],
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5,
                font=dict(size=11, color=_ct["legend"]), bgcolor="rgba(0,0,0,0)",
                bordercolor=_ct["grid"], borderwidth=1,
            ),
            margin=dict(l=10, r=10, t=50, b=80), height=500,
            hovermode="x unified",
        )
        st.plotly_chart(_fig_roc, width="stretch")
    else:
        if os.path.exists("outputs/roc_curves_all.png"):
            st.image("outputs/roc_curves_all.png", caption="ROC Curves — All Models")
        else:
            st.info("ROC curve data not available. Run the training notebook (03) to generate it.")

    # ────────────────────────────────────────────
    # PART 6 — CONFUSION MATRICES
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f522</span>'
        '<span class="section-title">Confusion Matrices</span>'
        '<span class="section-badge">Classification</span></div>',
        unsafe_allow_html=True,
    )

    if _dash_data and "confusion_matrices" in _dash_data and _dash_data["confusion_matrices"]:
        _cm_names = list(_dash_data["confusion_matrices"].keys())
        _cm_cols_count = min(len(_cm_names), 4)
        _cm_cols = st.columns(_cm_cols_count)

        _dark = st.session_state.get("dark_mode", True)
        _cm_text_color = "#f8fafc" if _dark else "#0f172a"
        _cm_colorscale = [
            [0.0, "#164e63" if _dark else "#e0f2fe"],
            [0.5, "#0891b2" if _dark else "#7dd3fc"],
            [1.0, "#22d3ee" if _dark else "#0284c7"],
        ]

        for _ci, _cm_name in enumerate(_cm_names):
            with _cm_cols[_ci % _cm_cols_count]:
                _cm = np.array(_dash_data["confusion_matrices"][_cm_name])
                _cm_labels = ["Default", "Repaid"]
                _is_best = (_cm_name == _dash_data.get("best_model_name", ""))
                _cm_text = [[str(v) for v in row] for row in _cm]

                _fig_cm = go.Figure(go.Heatmap(
                    z=_cm[::-1],
                    x=_cm_labels,
                    y=_cm_labels[::-1],
                    text=_cm_text[::-1],
                    texttemplate="%{text}",
                    textfont=dict(size=18, color=_cm_text_color),
                    colorscale=_cm_colorscale,
                    showscale=False,
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
                ))

                _cm_title = f"{'\U0001f3c6 ' if _is_best else ''}{_cm_name}"
                _fig_cm.update_layout(
                    title=dict(text=_cm_title,
                               font=dict(size=13, color=_ct["title"], family="Inter,sans-serif")),
                    xaxis=dict(
                        title=dict(text="Predicted", font=dict(color=_ct["axis"])),
                        tickfont=dict(size=11, color=_ct["text"]),
                        side="bottom",
                    ),
                    yaxis=dict(
                        title=dict(text="Actual", font=dict(color=_ct["axis"])),
                        tickfont=dict(size=11, color=_ct["text"]),
                    ),
                    plot_bgcolor=_ct["bg"], paper_bgcolor=_ct["bg"],
                    margin=dict(l=10, r=10, t=40, b=10), height=300,
                )
                st.plotly_chart(_fig_cm, width="stretch")
    else:
        if os.path.exists("outputs/confusion_matrices_all.png"):
            st.image("outputs/confusion_matrices_all.png", caption="Confusion Matrices — All Models")
        else:
            st.info("Confusion matrix data not available. Run the training notebook (03) to generate it.")

    # ────────────────────────────────────────────
    # PART 7 — FEATURE IMPORTANCE (BEST MODEL)
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4cc</span>'
        '<span class="section-title">Feature Importance — Best Model</span>'
        '<span class="section-badge">Explainability</span></div>',
        unsafe_allow_html=True,
    )

    _feat_imp_data = None
    if _dash_data and "feature_importance" in _dash_data and _dash_data["feature_importance"]:
        _feat_imp_data = _dash_data["feature_importance"]
    elif os.path.exists("outputs/feature_importance.csv"):
        _fi_csv = pd.read_csv("outputs/feature_importance.csv")
        if "Feature" in _fi_csv.columns and "Importance" in _fi_csv.columns:
            _feat_imp_data = dict(zip(_fi_csv["Feature"], _fi_csv["Importance"]))
        elif len(_fi_csv.columns) == 2:
            _feat_imp_data = dict(zip(_fi_csv.iloc[:, 0], _fi_csv.iloc[:, 1]))

    if _feat_imp_data:
        _fi_sorted = sorted(_feat_imp_data.items(), key=lambda x: abs(x[1]), reverse=True)
        _fi_top = _fi_sorted[:20]
        _fi_features = [f[0] for f in _fi_top][::-1]
        _fi_values = [f[1] for f in _fi_top][::-1]

        _fintech_gradient = [
            '#06b6d4', '#0ea5e9', '#3b82f6', '#6366f1', '#8b5cf6',
            '#a855f7', '#d946ef', '#ec4899', '#f43f5e', '#ef4444',
        ] * 2

        _fig_fi = go.Figure(go.Bar(
            y=_fi_features, x=_fi_values,
            orientation='h',
            marker=dict(
                color=_fintech_gradient[:len(_fi_features)],
                line=dict(width=0),
                cornerradius=4,
            ),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
        ))

        _fi_model_name = _dash_data.get("best_model_name", _best["Model"]) if _dash_data else _best["Model"]
        _fig_fi.update_layout(
            title=dict(
                text=f"Top {len(_fi_top)} Features — {_fi_model_name}",
                font=dict(size=15, color=_ct["title"], family="Inter,sans-serif"),
            ),
            plot_bgcolor=_ct["bg"], paper_bgcolor=_ct["bg"],
            xaxis=dict(
                title=dict(text="Importance Score", font=dict(color=_ct["axis"])),
                showgrid=True, gridcolor=_ct["grid"],
                tickfont=dict(size=11, color=_ct["axis"]),
            ),
            yaxis=dict(tickfont=dict(size=11, color=_ct["text"], family="Inter,sans-serif")),
            margin=dict(l=10, r=20, t=45, b=10), height=500,
            bargap=0.2,
        )
        st.plotly_chart(_fig_fi, width="stretch")
    else:
        st.info("Feature importance data not available. Run the training notebook (03) to generate it.")

    # ────────────────────────────────────────────
    # PART 8 — RADAR CHARTS
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f578️</span>'
        '<span class="section-title">Performance Radar</span>'
        '<span class="section-badge">Radar</span></div>',
        unsafe_allow_html=True,
    )

    _radar_metrics = [m for m in _metric_cols if m in _mc.columns]

    if len(_mc) >= 1 and len(_radar_metrics) >= 3:
        _radar_cols_count = min(len(_mc), 3)
        _radar_cols = st.columns(_radar_cols_count)

        for _ri, (_idx, _row) in enumerate(_mc.iterrows()):
            with _radar_cols[_ri % _radar_cols_count]:
                _vals = [_row[m] for m in _radar_metrics]
                _vals_closed = _vals + [_vals[0]]
                _theta_closed = _radar_metrics + [_radar_metrics[0]]
                _mclr = _model_palette[_ri % len(_model_palette)]
                _fill_r = int(_mclr[1:3], 16)
                _fill_g = int(_mclr[3:5], 16)
                _fill_b = int(_mclr[5:7], 16)

                _fig_ri = go.Figure()
                _fig_ri.add_trace(go.Scatterpolar(
                    r=_vals_closed, theta=_theta_closed,
                    fill="toself",
                    fillcolor=f"rgba({_fill_r},{_fill_g},{_fill_b},0.2)",
                    line=dict(color=_mclr, width=2.5),
                    marker=dict(size=6, color=_mclr),
                    name=_row["Model"],
                    hovertemplate=_row["Model"] + "<br>%{theta}: %{r:.4f}<extra></extra>",
                ))
                _is_best_r = (_row["Model"] == _orig_best_model)
                _fig_ri.update_layout(
                    polar=dict(
                        bgcolor=_ct["bg"],
                        radialaxis=dict(
                            visible=True, range=[0, 1.05], showline=False,
                            gridcolor=_ct["grid"],
                            tickfont=dict(size=8, color=_ct["muted"]),
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=10, color=_ct["text"], family="Inter,sans-serif"),
                            gridcolor=_ct["grid"],
                        ),
                    ),
                    title=dict(
                        text=f"{'\U0001f3c6 ' if _is_best_r else ''}{_row['Model']}",
                        font=dict(size=13, color=_ct["title"], family="Inter,sans-serif"),
                    ),
                    paper_bgcolor=_ct["bg"],
                    showlegend=False,
                    margin=dict(l=40, r=40, t=50, b=30), height=320,
                )
                st.plotly_chart(_fig_ri, width="stretch")

        st.markdown(
            '<div class="section-header" style="margin-top:12px;">'
            '<span class="section-icon">\U0001f504</span>'
            '<span class="section-title">Combined Radar Comparison</span></div>',
            unsafe_allow_html=True,
        )

        _fig_ro = go.Figure()
        for _ri, (_idx, _row) in enumerate(_df.iterrows()):
            _vals = [_row[m] for m in _radar_metrics]
            _vals_closed = _vals + [_vals[0]]
            _theta_closed = _radar_metrics + [_radar_metrics[0]]
            _mclr = _model_palette[_ri % len(_model_palette)]
            _fill_r = int(_mclr[1:3], 16)
            _fill_g = int(_mclr[3:5], 16)
            _fill_b = int(_mclr[5:7], 16)
            _fig_ro.add_trace(go.Scatterpolar(
                r=_vals_closed, theta=_theta_closed,
                fill="toself",
                fillcolor=f"rgba({_fill_r},{_fill_g},{_fill_b},0.12)",
                line=dict(color=_mclr, width=2.5),
                marker=dict(size=6, color=_mclr),
                name=_row["Model"],
                hovertemplate=_row["Model"] + "<br>%{theta}: %{r:.4f}<extra></extra>",
            ))
        _fig_ro.update_layout(
            polar=dict(
                bgcolor=_ct["bg"],
                radialaxis=dict(
                    visible=True, range=[0, 1.05], showline=False,
                    gridcolor=_ct["grid"],
                    tickfont=dict(size=9, color=_ct["muted"]),
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color=_ct["text"], family="Inter,sans-serif"),
                    gridcolor=_ct["grid"],
                ),
            ),
            title=dict(
                text="All Models — Metric Radar",
                font=dict(size=15, color=_ct["title"], family="Inter,sans-serif"),
            ),
            paper_bgcolor=_ct["bg"],
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5,
                font=dict(size=12, color=_ct["legend"]),
            ),
            margin=dict(l=50, r=50, t=50, b=60), height=450,
        )
        st.plotly_chart(_fig_ro, width="stretch")

    # ────────────────────────────────────────────
    # PART 9 — AI INSIGHT ENGINE
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4a1</span>'
        '<span class="section-title">AI Performance Insights</span>'
        '<span class="section-badge">AI Analysis</span></div>',
        unsafe_allow_html=True,
    )

    _insights = []

    _sel_top = None
    try:
        if _sel_metric in _mc.columns and _mc[_sel_metric].notna().any():
            _sel_top = _mc.loc[_mc[_sel_metric].idxmax()]
            _insights.append(
                f"\U0001f3af **{_sel_top['Model']}** dominates in **{_sel_metric}** "
                f"with a score of **{_sel_top[_sel_metric]:.4f}**."
            )
    except (ValueError, KeyError):
        pass

    _ensemble = [n for n in _mc["Model"] if any(k in n for k in ["Random Forest", "XGBoost", "Gradient"])]
    _non_ensemble = [n for n in _mc["Model"] if n not in _ensemble]
    if _ensemble and _non_ensemble and "Accuracy" in _mc.columns:
        _ens_avg = _mc[_mc["Model"].isin(_ensemble)]["Accuracy"].mean()
        _ne_avg = _mc[_mc["Model"].isin(_non_ensemble)]["Accuracy"].mean()
        if _ens_avg > _ne_avg:
            _insights.append(
                f"\U0001f4c8 Ensemble models average **{_ens_avg:.4f}** accuracy vs "
                f"**{_ne_avg:.4f}** for baseline models — a "
                f"**{(_ens_avg - _ne_avg)*100:.1f}pp** improvement."
            )

    for _m in _metric_cols:
        if _m == _sel_metric or _m not in _mc.columns:
            continue
        try:
            _t = _mc.loc[_mc[_m].idxmax()]
            if _sel_top is None or _t["Model"] != _sel_top["Model"]:
                _insights.append(
                    f"\U0001f4ca **{_t['Model']}** leads in **{_m}** ({_t[_m]:.4f}), "
                    f"showing strength in a different dimension."
                )
        except (ValueError, KeyError):
            continue

    if "F1 Score" in _mc.columns and "Precision" in _mc.columns and "Recall" in _mc.columns:
        for _, _r in _mc.iterrows():
            _gap = abs(_r["Precision"] - _r["Recall"])
            if _gap < 0.02:
                _insights.append(
                    f"⚖️ **{_r['Model']}** shows excellent precision-recall balance "
                    f"(gap: {_gap:.3f})."
                )

    if len(_metric_cols) >= 3 and len(_mc) > 1:
        try:
            _stds = _mc[_metric_cols].std(axis=1)
            _most_stable_idx = _stds.idxmin()
            _most_stable = _mc.loc[_most_stable_idx]
            _insights.append(
                f"\U0001f512 **{_most_stable['Model']}** is the most stable model "
                f"(cross-metric std: {_stds[_most_stable_idx]:.4f})."
            )
        except (ValueError, KeyError):
            pass

    if "Recall" in _mc.columns:
        try:
            _recall_leader = _mc.loc[_mc["Recall"].idxmax()]
            if _sel_top is None or _recall_leader["Model"] != _sel_top["Model"]:
                _insights.append(
                    f"\U0001f50d **{_recall_leader['Model']}** has the highest recall "
                    f"(**{_recall_leader['Recall']:.4f}**) — best at catching actual defaults."
                )
        except (ValueError, KeyError):
            pass

    _spread = _mc[_sel_metric].max() - _mc[_sel_metric].min() if _sel_metric in _mc.columns else 0
    if _spread < 0.03 and len(_mc) > 2:
        _insights.append(
            f"\U0001f52c All models score within **{_spread:.3f}** of each other on {_sel_metric} — "
            f"consider secondary metrics for final selection."
        )

    try:
        if _sel_metric in _mc.columns and _mc[_sel_metric].notna().any():
            _worst = _mc.loc[_mc[_sel_metric].idxmin()]
            if _worst[_sel_metric] < _mc[_sel_metric].max() - 0.02:
                _insights.append(
                    f"⚠️ **{_worst['Model']}** underperforms on {_sel_metric} "
                    f"({_worst[_sel_metric]:.4f}), suggesting lower generalization."
                )
    except (ValueError, KeyError):
        pass

    st.markdown(
        '<div style="background:var(--lp-panel-bg);'
        'border-radius:14px;padding:22px 26px;margin:8px 0 18px 0;'
        'border:1px solid var(--lp-panel-border);'
        'box-shadow:var(--lp-panel-shadow);">'
        '<div style="color:var(--lp-text-3);font-size:0.7rem;text-transform:uppercase;'
        'letter-spacing:1.2px;font-weight:600;margin-bottom:12px;">'
        f'Insights for {_sel_metric}</div>'
        + "".join(
            f'<div style="color:var(--lp-text-2);font-size:0.88rem;line-height:1.8;'
            f'padding:3px 0;">{ins}</div>' for ins in _insights
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────
    # PART 10 — PRODUCTION RECOMMENDATION
    # ────────────────────────────────────────────
    _reasons = []
    for _m in _metric_cols:
        try:
            if _m in _mc.columns and _mc[_m].notna().any() and _mc[_m].idxmax() == _best_idx:
                _reasons.append(f"Highest {_m}")
        except (ValueError, KeyError):
            continue
    if not _reasons:
        _reasons.append("Best overall weighted score")

    if "Precision" in _mc.columns and "Recall" in _mc.columns:
        _best_gap = abs(_best.get("Precision", 0) - _best.get("Recall", 0))
        if _best_gap < 0.03:
            _reasons.append(f"Balanced Precision & Recall (gap: {_best_gap:.3f})")

    _rank_lines = ""
    for _m in _metric_cols:
        if _m in _mc.columns:
            _rank_val = int(_mc[_m].rank(ascending=False).loc[_best_idx])
            _rank_lines += (
                f'<div style="display:flex;justify-content:space-between;padding:4px 0;'
                f'border-bottom:1px solid rgba(148,163,184,0.1);">'
                f'<span style="color:var(--lp-text-2);font-size:0.82rem;">{_m}</span>'
                f'<span style="color:{_palette.get(_m, "#fff")};font-weight:700;font-size:0.82rem;">'
                f'{_best.get(_m, 0):.4f} (Rank #{_rank_val})</span></div>'
            )

    st.markdown(
        f"""<div style="background:var(--lp-rec-bg);
        border-radius:14px;padding:24px 28px;margin:10px 0 20px 0;
        border-left:4px solid #2f7cff;
        box-shadow:var(--lp-panel-shadow);">
        <div style="color:var(--lp-text-3);font-size:0.75rem;text-transform:uppercase;
        letter-spacing:1.5px;font-weight:600;">Recommended Production Model</div>
        <div style="color:var(--lp-text);font-size:1.35rem;font-weight:700;
        margin:6px 0 14px 0;">{_best['Model']}</div>
        <div style="color:var(--lp-text-2);font-size:0.85rem;line-height:1.7;margin-bottom:14px;">
        {'<br>'.join('✅ ' + r for r in _reasons)}<br>
        ✅ Best Generalization</div>
        <div style="color:var(--lp-text-3);font-size:0.68rem;text-transform:uppercase;
        letter-spacing:1px;font-weight:600;margin-bottom:8px;">Metric Breakdown</div>
        {_rank_lines}
        </div>""",
        unsafe_allow_html=True,
    )

    # ────────────────────────────────────────────
    # PART 11 — EXPORT
    # ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-header">'
        '<span class="section-icon">\U0001f4e5</span>'
        '<span class="section-title">Export</span>'
        '<span class="section-badge">Download</span></div>',
        unsafe_allow_html=True,
    )
    _export = _mc.copy()
    for _c in _metric_cols:
        if _c in _export.columns:
            _export[_c] = _export[_c].round(4)
    _export["Best Model"] = _export["Model"].apply(lambda x: "⭐" if x == _orig_best_model else "")
    st.download_button(
        label="\U0001f4e5 Download Model Comparison Report (CSV)",
        data=_export.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_report.csv",
        mime="text/csv",
        key="dash_export_csv",
    )

else:
    st.info(
        "\U0001f4ca Model comparison data not found. Run the training notebook (03) to generate "
        "`outputs/model_comparison.csv` and unlock the full dashboard."
    )

# =========================
# AI FINANCIAL ASSISTANT CHATBOT
# =========================
st.markdown("---")

with st.expander("🤖 AI Financial Assistant — Click to open chat", expanded=False):
    chat_col1, chat_col2 = st.columns([3, 1])
    with chat_col1:
        st.markdown("##### 💬 Ask the AI Loan Assistant")
    with chat_col2:
        if st.button("🗑️ Clear Chat", key="clear_chat_button"):
            st.session_state.chat_history = []
            st.rerun()

    st.info(
        "**Suggested Questions:** What is my loan status? • Why was I approved? • How can I improve? • Explain my credit score."
    )

    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                continue
            with st.chat_message(msg["role"], avatar="🏦" if msg["role"] == "assistant" else "👤"):
                st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask about loan status, risk factors, or improvement strategy...")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        assistant_reply = loan_chatbot_response(
            user_prompt,
            st.session_state.last_applicant_data,
            st.session_state.last_prediction,
            st.session_state.last_probability,
        )

        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
        st.rerun()