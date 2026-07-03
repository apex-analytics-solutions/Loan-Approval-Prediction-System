"""
Production-Ready Prediction Engine — Loan Prediction System.

Guarantees exact preprocessing parity with train.py:
  raw input → validate → LabelEncode categoricals → log-transforms → feature-order align → predict

Usage:
    from predict_engine import predict, PredictionResult
    result = predict(raw_input_dict, model, encoders, feature_names)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

log = logging.getLogger("loan_project.predict")


# ─────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PredictionResult:
    prediction: str        # "Approved" | "Rejected"
    probability: float     # 0.0 – 1.0  (probability of class 1 = repaid)
    risk_level: str        # "Low" | "Medium" | "High"
    risk_color: str        # hex color for UI
    risk_emoji: str        # 🟢 🟡 🔴
    confidence: str        # "High" | "Medium" | "Low"
    raw_pred: int          # 0 or 1
    input_vector: object   # pd.DataFrame — the exact row fed to the model


# ─────────────────────────────────────────────
# Validation Schema
# ─────────────────────────────────────────────

FIELD_SCHEMA: Dict[str, dict] = {
    "age":                    {"type": "int",   "min": 18,   "max": 100,       "label": "Age"},
    "gender":                 {"type": "cat",                                   "label": "Gender"},
    "marital_status":         {"type": "cat",                                   "label": "Marital Status"},
    "education_level":        {"type": "cat",                                   "label": "Education Level"},
    "employment_status":      {"type": "cat",                                   "label": "Employment Status"},
    "annual_income":          {"type": "float", "min": 0,    "max": 10_000_000, "label": "Annual Income ($)"},
    "monthly_income":         {"type": "float", "min": 0,    "max": 1_000_000,  "label": "Monthly Income ($)"},
    "credit_score":           {"type": "int",   "min": 300,  "max": 850,        "label": "Credit Score"},
    "debt_to_income_ratio":   {"type": "float", "min": 0.0,  "max": 1.0,        "label": "Debt-to-Income Ratio"},
    "loan_amount":            {"type": "float", "min": 100,  "max": 10_000_000, "label": "Loan Amount ($)"},
    "loan_purpose":           {"type": "cat",                                   "label": "Loan Purpose"},
    "grade_subgrade":         {"type": "cat",                                   "label": "Grade / Subgrade"},
    "interest_rate":          {"type": "float", "min": 0.1,  "max": 35.0,       "label": "Interest Rate (%)"},
    "loan_term":              {"type": "choice", "options": [36, 60],            "label": "Loan Term (months)"},
    "installment":            {"type": "float", "min": 1,    "max": 100_000,    "label": "Monthly Installment ($)"},
    "num_of_open_accounts":   {"type": "int",   "min": 0,    "max": 50,         "label": "Open Accounts"},
    "total_credit_limit":     {"type": "float", "min": 0,    "max": 10_000_000, "label": "Total Credit Limit ($)"},
    "current_balance":        {"type": "float", "min": 0,    "max": 10_000_000, "label": "Current Balance ($)"},
    "delinquency_history":    {"type": "int",   "min": 0,    "max": 20,         "label": "Delinquency History"},
    "public_records":         {"type": "int",   "min": 0,    "max": 10,         "label": "Public Records"},
    "num_of_delinquencies":   {"type": "int",   "min": 0,    "max": 20,         "label": "Number of Delinquencies"},
}

CATEGORICAL_COLS = [
    "gender", "marital_status", "education_level",
    "employment_status", "loan_purpose", "grade_subgrade",
]

NUMERIC_PASSTHROUGH = [
    "age", "annual_income", "debt_to_income_ratio", "credit_score",
    "interest_rate", "loan_term", "installment",
    "num_of_open_accounts", "total_credit_limit", "current_balance",
    "delinquency_history", "public_records", "num_of_delinquencies",
]

ENGINEERED_FEATURES = {
    "monthly_income_log": "monthly_income",
    "loan_amount_log": "loan_amount",
}


# ─────────────────────────────────────────────
# Feature Alignment Guard
# ─────────────────────────────────────────────

def validate_feature_alignment(feature_names: List[str]) -> None:
    """
    Verify that CATEGORICAL_COLS + NUMERIC_PASSTHROUGH + ENGINEERED_FEATURES
    covers exactly the set of features the model expects.
    Raises ValueError on mismatch so config drift is caught at startup.
    """
    expected = set(CATEGORICAL_COLS) | set(NUMERIC_PASSTHROUGH) | set(ENGINEERED_FEATURES)
    actual = set(feature_names)

    missing_from_pipeline = actual - expected
    extra_in_pipeline = expected - actual

    errors = []
    if missing_from_pipeline:
        errors.append(
            f"Model expects features not covered by the pipeline: {sorted(missing_from_pipeline)}"
        )
    if extra_in_pipeline:
        errors.append(
            f"Pipeline defines features the model doesn't use: {sorted(extra_in_pipeline)}"
        )
    if errors:
        raise ValueError(
            "Feature alignment mismatch between predict_engine and model artifact:\n"
            + "\n".join(f"  • {e}" for e in errors)
        )


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────

def validate_input(
    raw: Dict[str, Any],
    encoders: dict,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate every field in *raw* against FIELD_SCHEMA.
    Returns (cleaned_values, list_of_error_strings).
    """
    errors: List[str] = []
    cleaned: Dict[str, Any] = {}

    for field, schema in FIELD_SCHEMA.items():
        val = raw.get(field)
        label = schema["label"]

        if val is None:
            errors.append(f"{label} is required.")
            continue

        ftype = schema["type"]

        # ── categorical ──
        if ftype == "cat":
            if field in encoders:
                valid = list(encoders[field].classes_)
                if val not in valid:
                    errors.append(f"{label}: '{val}' is not a valid option.")
                    continue
            cleaned[field] = val
            continue

        # ── choice (fixed set of numeric values) ──
        if ftype == "choice":
            try:
                val = int(val)
            except (ValueError, TypeError):
                errors.append(f"{label} must be an integer.")
                continue
            if val not in schema["options"]:
                errors.append(f"{label} must be one of {schema['options']}.")
                continue
            cleaned[field] = val
            continue

        # ── int / float ──
        try:
            val = int(val) if ftype == "int" else float(val)
        except (ValueError, TypeError):
            errors.append(f"{label} must be {'an integer' if ftype == 'int' else 'a number'}.")
            continue

        lo, hi = schema.get("min"), schema.get("max")
        if lo is not None and val < lo:
            errors.append(f"{label} cannot be less than {lo}.")
            continue
        if hi is not None and val > hi:
            errors.append(f"{label} cannot exceed {hi}.")
            continue

        cleaned[field] = val

    return cleaned, errors


# ─────────────────────────────────────────────
# Preprocessing  (mirrors train.py exactly)
# ─────────────────────────────────────────────

def preprocess(
    cleaned: Dict[str, Any],
    encoders: dict,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Transform validated raw inputs into the exact feature vector the model expects.

    Pipeline (same as train.py):
        1. LabelEncode each categorical column with the SAME fitted encoder
        2. Pass numeric columns through unchanged
        3. Compute monthly_income_log = log1p(monthly_income)
        4. Compute loan_amount_log    = log1p(loan_amount)
        5. Assemble in the exact column order stored in feature_names
    """
    row: Dict[str, float] = {}

    # 1  Categoricals → LabelEncoded ints
    for col in CATEGORICAL_COLS:
        val = cleaned[col]
        le = encoders[col]
        known = set(le.classes_)
        if val not in known:
            raise ValueError(
                f"Unseen category '{val}' for feature '{col}'. "
                f"Valid values: {sorted(known)}"
            )
        row[col] = float(le.transform([val])[0])

    # 2  Numeric pass-through
    for col in NUMERIC_PASSTHROUGH:
        row[col] = float(cleaned[col])

    # 3  Engineered features (must match train.py)
    for eng_col, source_col in ENGINEERED_FEATURES.items():
        row[eng_col] = float(np.log1p(cleaned[source_col]))

    # 4  Assemble in exact feature order — fail on missing rather than silent zero-fill
    missing = [f for f in feature_names if f not in row]
    if missing:
        raise RuntimeError(
            f"Preprocessing produced no value for feature(s): {missing}. "
            f"Check CATEGORICAL_COLS, NUMERIC_PASSTHROUGH, and ENGINEERED_FEATURES."
        )
    vector = {f: row[f] for f in feature_names}
    return pd.DataFrame([vector])[feature_names]


# ─────────────────────────────────────────────
# Risk Classification
# ─────────────────────────────────────────────

def _classify_risk(proba: float) -> Tuple[str, str, str]:
    """Returns (risk_level, risk_color, risk_emoji)."""
    if proba >= 0.75:
        return "Low", "#10b981", "🟢"
    if proba >= 0.50:
        return "Medium", "#f59e0b", "🟡"
    return "High", "#ef4444", "🔴"


def _classify_confidence(proba: float) -> str:
    distance = abs(proba - 0.5)
    if distance > 0.25:
        return "High"
    if distance > 0.10:
        return "Medium"
    return "Low"


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def predict(
    raw_input: Dict[str, Any],
    model,
    encoders: dict,
    feature_names: List[str],
) -> PredictionResult:
    """
    Full prediction pipeline:  validate → preprocess → predict → classify.

    Raises
    ------
    ValueError  if any input field fails validation (message lists all errors).
    RuntimeError  if the model inference itself fails.
    """
    # ── Step 1: Validate ──
    cleaned, errors = validate_input(raw_input, encoders)
    if errors:
        log.warning("Validation failed: %d errors", len(errors))
        raise ValueError(
            "Input validation failed:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    # ── Step 2: Preprocess ──
    input_df = preprocess(cleaned, encoders, feature_names)
    log.debug("Input vector shape: %s", input_df.shape)

    # ── Step 3: Model inference (single call) ──
    try:
        proba_array = model.predict_proba(input_df)[0]
        proba = float(proba_array[1])
        raw_pred = int(proba >= 0.5)
    except Exception as exc:
        log.error("Model inference failed: %s", exc)
        raise RuntimeError(f"Model inference failed: {exc}") from exc

    # ── Step 4: Classify ──
    risk_level, risk_color, risk_emoji = _classify_risk(proba)
    confidence = _classify_confidence(proba)
    prediction = "Approved" if raw_pred == 1 else "Rejected"

    log.info(
        "Prediction: %s | prob=%.4f | risk=%s | confidence=%s",
        prediction, proba, risk_level, confidence,
    )

    return PredictionResult(
        prediction=prediction,
        probability=proba,
        risk_level=risk_level,
        risk_color=risk_color,
        risk_emoji=risk_emoji,
        confidence=confidence,
        raw_pred=raw_pred,
        input_vector=input_df,
    )
