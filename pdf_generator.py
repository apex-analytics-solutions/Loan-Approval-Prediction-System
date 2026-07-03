"""Premium multi-page fintech PDF report generator for AI loan assessment."""

from __future__ import annotations

import csv
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------
# Core normalization and extraction helpers.
# ---------------------------------------------------------

def _to_dict(data: Any) -> Dict[str, Any]:
    """Convert incoming applicant data into a dictionary safely."""
    if isinstance(data, dict):
        return data
    if hasattr(data, "to_dict"):
        converted = data.to_dict()
        if isinstance(converted, dict):
            return converted
    raise TypeError("applicant_data must be a dict-like object")


def _normalize_probability(probability: Any) -> float:
    """Normalize probability into a 0-1 float range."""
    value = float(probability)
    if value > 1:
        value = value / 100.0
    return max(0.0, min(1.0, value))


def _normalize_status(prediction: Any) -> str:
    """Normalize prediction into APPROVED or REJECTED text."""
    if isinstance(prediction, str):
        normalized = prediction.strip().lower()
        if normalized in {"approved", "approve", "1", "true", "yes"}:
            return "APPROVED"
        if normalized in {"rejected", "reject", "0", "false", "no"}:
            return "REJECTED"
    if isinstance(prediction, (int, float, bool)):
        return "APPROVED" if int(prediction) == 1 else "REJECTED"
    return "APPROVED" if bool(prediction) else "REJECTED"


def _extract_value(data: Dict[str, Any], keys: Iterable[str], default: str = "N/A") -> Any:
    """Fetch the first available value from multiple possible keys."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _safe_float(value: Any) -> Optional[float]:
    """Convert value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------
# Visual style system for premium fintech report design.
# ---------------------------------------------------------

def _build_styles() -> StyleSheet1:
    """Create consistent premium report typography styles."""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="BankBrand",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            textColor=colors.white,
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BrandSub",
            parent=styles["Heading2"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.white,
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#0F172A"),
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSubtitle",
            parent=styles["Heading2"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#334155"),
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitleBlue",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1D4ED8"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextReport",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#0F172A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextMuted",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748B"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="VerdictText",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#1E3A8A"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="FooterCenter",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=1,
        )
    )
    styles.add(
        ParagraphStyle(
            name="StatusText",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=14,
            textColor=colors.white,
            alignment=1,
        )
    )
    return styles


def _draw_page_number(canvas, document) -> None:
    """Draw page number on every page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawRightString(A4[0] - document.rightMargin, 8 * mm, f"Page {document.page}")
    canvas.restoreState()


def _bank_brand_header(styles: StyleSheet1) -> Table:
    """Create premium bank branding header with subtle blue gradient effect."""
    header = Table(
        [
            [Paragraph("GLOBAL AI FINTECH BANK", styles["BankBrand"])],
            [Paragraph("Loan Risk Intelligence Report", styles["BrandSub"])],
        ],
        colWidths=[170 * mm],
    )
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#2563EB")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#1D4ED8")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return header


def _section_band(title: str, styles: StyleSheet1) -> Table:
    """Create blue section title band."""
    band = Table([[Paragraph(title, styles["SectionTitleBlue"])]], colWidths=[170 * mm])
    band.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF1FF")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#BFDBFE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return band


# ---------------------------------------------------------
# Financial score gauge and interpretation.
# ---------------------------------------------------------

def _risk_interpretation(score: int) -> str:
    """Return interpretation label for risk score."""
    if score <= 30:
        return "Low Risk"
    if score <= 60:
        return "Medium Risk"
    return "High Risk"


def _text_gauge(score: int) -> str:
    """Create a text-based gauge bar using unicode blocks."""
    filled = max(0, min(10, round(score / 10)))
    return f"{'█' * filled}{'░' * (10 - filled)} {score}/100"


def _approval_gauge(probability: float) -> str:
    """Create text-based approval probability gauge."""
    score = round(probability * 100)
    filled = max(0, min(10, round(score / 10)))
    return f"{'█' * filled}{'░' * (10 - filled)} {score}%"


# ---------------------------------------------------------
# Mandatory modular functions requested by user.
# ---------------------------------------------------------

def create_cover_page(
    story: List[Any], styles: StyleSheet1, generated_at: str, model_name: str, applicant: Dict[str, Any]
) -> None:
    """Create premium cover page with bank branding and professional footer."""
    story.append(_bank_brand_header(styles))
    story.append(Spacer(1, 16 * mm))
    story.append(Paragraph("AI Loan Eligibility Prediction System", styles["CoverTitle"]))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Machine Learning Risk Assessment Report", styles["CoverSubtitle"]))
    story.append(Spacer(1, 10 * mm))

    app_name = _extract_value(applicant, ["Full Name", "full_name", "Name"])
    app_id = _extract_value(applicant, ["Application ID", "application_id", "App ID"])

    cover_meta = Table(
        [
            ["Applicant Name", str(app_name)],
            ["Application ID", str(app_id)],
            ["Generated Date & Time", generated_at],
            ["Best Machine Learning Model Used", model_name],
            ["Applicant Loan Assessment", "Comprehensive AI credit risk evaluation"],
        ],
        colWidths=[65 * mm, 105 * mm],
        hAlign="LEFT",
    )
    cover_meta.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(cover_meta)
    story.append(Spacer(1, 145 * mm))
    story.append(Paragraph("Generated using Python, Scikit-Learn, Streamlit", styles["FooterCenter"]))
    story.append(PageBreak())


def create_summary_page(
    story: List[Any],
    styles: StyleSheet1,
    status: str,
    probability: float,
    risk_score: int,
) -> None:
    """Create executive summary page with risk/decision overview and gauges."""
    story.append(_section_band("Executive Summary", styles))
    story.append(Spacer(1, 5 * mm))

    verdict = (
        "Applicant shows strong financial stability with low predicted credit risk."
        if status == "APPROVED"
        else "Applicant shows moderate financial stability with controlled risk exposure."
        if risk_score <= 60
        else "Applicant shows elevated financial risk exposure requiring corrective financial actions."
    )

    summary_table = Table(
        [
            ["Applicant Risk Summary", _risk_interpretation(risk_score)],
            ["AI Decision Summary", status],
            ["Financial Stability Score (0-100)", str(max(0, 100 - risk_score))],
            ["Approval Probability Gauge", _approval_gauge(probability)],
            ["One-line AI Verdict", verdict],
        ],
        colWidths=[65 * mm, 105 * mm],
        hAlign="LEFT",
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    story.append(summary_table)
    story.append(Spacer(1, 5 * mm))
    story.append(
        Paragraph(f"Risk Score: {_text_gauge(risk_score)}", styles["BodyTextReport"])
    )
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph("Risk Interpretation: 0-30 Low Risk | 31-60 Medium Risk | 61-100 High Risk", styles["BodyTextMuted"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(verdict, styles["VerdictText"]))
    story.append(PageBreak())


def _applicant_identity_rows(applicant: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return applicant identity/contact fields."""
    return [
        ("Full Name", _extract_value(applicant, ["Full Name", "full_name", "Name"])),
        ("Application ID", _extract_value(applicant, ["Application ID", "application_id", "App ID"])),
        ("CNIC / National ID", _extract_value(applicant, ["CNIC", "cnic", "CNIC / National ID", "national_id"])),
        ("Phone Number", _extract_value(applicant, ["Phone", "phone", "Phone Number", "phone_number"])),
        ("Email Address", _extract_value(applicant, ["Email", "email", "Email Address", "email_address"])),
    ]


def _applicant_personal_rows(applicant: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return applicant personal detail fields."""
    return [
        ("Age", _extract_value(applicant, ["Age", "age"])),
        ("Gender", _extract_value(applicant, ["Gender", "gender"])),
        ("Marital Status", _extract_value(applicant, ["Marital Status", "marital_status"])),
        ("Education Level", _extract_value(applicant, ["Education Level", "education_level"])),
        ("Employment Status", _extract_value(applicant, ["Employment Status", "employment_status"])),
    ]


def _applicant_financial_rows(applicant: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return applicant financial detail fields."""
    annual = _extract_value(applicant, ["Annual Income", "annual_income"])
    monthly = _extract_value(applicant, ["Monthly Income", "monthly_income"])
    credit = _extract_value(applicant, ["Credit Score", "credit_score"])
    dti = _extract_value(applicant, ["Debt To Income Ratio", "debt_to_income_ratio", "debt_ratio"])

    annual_fmt = f"${float(annual):,.2f}" if _safe_float(annual) is not None else annual
    monthly_fmt = f"${float(monthly):,.2f}" if _safe_float(monthly) is not None else monthly
    dti_val = _safe_float(dti)
    dti_fmt = f"{dti_val * 100:.1f}%" if dti_val is not None else dti

    return [
        ("Annual Income", annual_fmt),
        ("Monthly Income", monthly_fmt),
        ("Credit Score", credit),
        ("Debt To Income Ratio", dti_fmt),
    ]


def _applicant_loan_rows(applicant: Dict[str, Any]) -> List[Tuple[str, Any]]:
    """Return loan detail fields."""
    amount = _extract_value(applicant, ["Loan Amount", "loan_amount"])
    amount_fmt = f"${float(amount):,.2f}" if _safe_float(amount) is not None else amount

    rate = _extract_value(applicant, ["Interest Rate", "interest_rate"])
    rate_val = _safe_float(rate)
    rate_fmt = f"{rate_val:.1f}%" if rate_val is not None else rate

    return [
        ("Loan Amount", amount_fmt),
        ("Loan Purpose", _extract_value(applicant, ["Loan Purpose", "loan_purpose"])),
        ("Grade / Subgrade", _extract_value(applicant, ["Grade / Subgrade", "grade_subgrade", "Grade"])),
        ("Interest Rate", rate_fmt),
        ("Loan Term", _extract_value(applicant, ["Loan Term", "loan_term"])),
    ]


def _build_sectioned_table(
    sections: List[Tuple[str, List[Tuple[str, Any]]]],
) -> List[List[str]]:
    """Build table data with section headers and field rows."""
    data = [["Field", "Value"]]
    for section_title, rows in sections:
        data.append([section_title, ""])
        for key, value in rows:
            data.append([str(key), str(value)])
    return data


def create_applicant_table(story: List[Any], styles: StyleSheet1, applicant: Dict[str, Any]) -> None:
    """Create applicant profile page with sectioned layout and alternating row colors."""
    story.append(_section_band("Applicant Profile", styles))
    story.append(Spacer(1, 4 * mm))

    sections = [
        ("  Applicant Information", _applicant_identity_rows(applicant)),
        ("  Personal Details", _applicant_personal_rows(applicant)),
        ("  Financial Details", _applicant_financial_rows(applicant)),
        ("  Loan Details", _applicant_loan_rows(applicant)),
    ]

    data = _build_sectioned_table(sections)

    table = Table(data, colWidths=[62 * mm, 108 * mm], hAlign="LEFT")
    style_cmds = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]

    section_header_indices = []
    row_idx = 1
    for section_title, rows in sections:
        section_header_indices.append(row_idx)
        row_idx += 1 + len(rows)

    for idx in section_header_indices:
        if idx < len(data):
            style_cmds.extend([
                ("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#E0E7FF")),
                ("TEXTCOLOR", (0, idx), (-1, idx), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, idx), (-1, idx), "Helvetica-Bold"),
                ("FONTSIZE", (0, idx), (-1, idx), 10),
                ("SPAN", (0, idx), (-1, idx)),
            ])

    data_row = 0
    for idx in range(1, len(data)):
        if idx in section_header_indices:
            data_row = 0
            continue
        data_row += 1
        bg = colors.HexColor("#F8FAFC") if data_row % 2 == 0 else colors.white
        style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), bg))

    table.setStyle(TableStyle(style_cmds))
    story.append(table)
    story.append(PageBreak())


# ---------------------------------------------------------
# Decision and AI insight sections.
# ---------------------------------------------------------

def _status_color(status: str, risk_level: str) -> colors.Color:
    """Select status color for decision box."""
    if str(risk_level).strip().lower() == "medium":
        return colors.HexColor("#F59E0B")
    return colors.HexColor("#16A34A") if status == "APPROVED" else colors.HexColor("#DC2626")


def _build_decision_table(
    status: str,
    probability: float,
    risk_score: int,
    risk_level: str,
    model_name: str,
    prediction_date: str,
) -> Table:
    """Build AI decision table."""
    data = [
        ["Loan Status", status],
        ["Approval Probability", f"{probability * 100:.2f}%"],
        ["Financial Risk Score (0-100)", str(risk_score)],
        ["Risk Level", risk_level],
        ["Model Used", model_name],
        ["Prediction Date", prediction_date],
    ]

    table = Table(data, colWidths=[65 * mm, 105 * mm], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _ai_insights(status: str) -> List[str]:
    """Generate AI insights in a bank credit officer tone."""
    if status == "APPROVED":
        return [
            "Income-to-obligation balance indicates strong repayment readiness.",
            "Credit profile reflects disciplined payment behavior and healthy utilization.",
            "Debt exposure remains within acceptable underwriting thresholds.",
            "Employment and financial stability support long-term repayment consistency.",
            "Overall borrower profile aligns with prudent bank lending standards.",
        ]
    return [
        "Current leverage indicates elevated probability of repayment stress.",
        "Credit profile requires improvement before favorable lending terms are likely.",
        "Debt-to-income pressure should be reduced through liability optimization.",
        "Household cash flow resilience should be strengthened prior to reapplication.",
        "A phased credit recovery roadmap is recommended over the next 3-6 months.",
    ]


# ---------------------------------------------------------
# Visual analytics and model comparison helpers.
# ---------------------------------------------------------

def _image_specs(outputs_dir: Path) -> List[Tuple[Path, str, str]]:
    """Define visual analytics image files and captions."""
    return [
        (
            outputs_dir / "feature_importance.png",
            "Feature Importance",
            "Relative influence of key applicant attributes on model decision.",
        ),
        (
            outputs_dir / "confusion_matrix.png",
            "Confusion Matrix",
            "Distribution of true vs predicted classifications.",
        ),
        (
            outputs_dir / "roc_curve.png",
            "ROC Curve",
            "Model discrimination performance across threshold settings.",
        ),
        (
            outputs_dir / "model_dashboard.png",
            "Model Dashboard",
            "Consolidated overview of model diagnostics and performance.",
        ),
        (
            outputs_dir / "accuracy_comparison.png",
            "Accuracy Comparison",
            "Comparative accuracy across evaluated machine learning models.",
        ),
    ]


def _fit_image(path: Path, max_width: float, max_height: float) -> Optional[Image]:
    """Create centered scaled image safely, returning None if invalid."""
    try:
        reader = ImageReader(str(path))
        width, height = reader.getSize()
    except Exception:
        return None

    if width <= 0 or height <= 0:
        return None

    ratio = min(max_width / float(width), max_height / float(height), 1.0)
    img = Image(str(path), width=width * ratio, height=height * ratio)
    img.hAlign = "CENTER"
    return img


def create_visual_section(story: List[Any], styles: StyleSheet1, outputs_dir: Path) -> None:
    """Create advanced visual section with framed image cards and safe fallbacks."""
    story.append(_section_band("AI Visual Analytics", styles))
    story.append(Spacer(1, 4 * mm))

    for image_path, title, caption in _image_specs(outputs_dir):
        story.append(Paragraph(title, styles["SectionTitleBlue"]))

        if image_path.exists():
            image_flowable = _fit_image(image_path, max_width=160 * mm, max_height=86 * mm)
            if image_flowable is not None:
                image_box = Table([[image_flowable]], colWidths=[170 * mm], hAlign="CENTER")
                image_box.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#CBD5E1")),
                            ("LEFTPADDING", (0, 0), (-1, -1), 6),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 6),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ]
                    )
                )
                story.append(image_box)
            else:
                story.append(Paragraph("Data not available", styles["BodyTextMuted"]))
        else:
            story.append(Paragraph("Data not available", styles["BodyTextMuted"]))

        story.append(Spacer(1, 1.5 * mm))
        story.append(Paragraph(caption, styles["BodyTextMuted"]))
        story.append(Spacer(1, 4.5 * mm))

    story.append(PageBreak())


def _read_model_comparison(csv_path: Path) -> List[Dict[str, Any]]:
    """Read and normalize model comparison CSV rows."""
    if not csv_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    required = ["Model", "Accuracy", "Precision", "Recall", "F1 Score"]

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            return []

        mapper = {name.strip().lower(): name for name in reader.fieldnames}
        for raw in reader:
            normalized: Dict[str, Any] = {}
            for col in required:
                source = mapper.get(col.lower())
                normalized[col] = raw.get(source, "N/A") if source else "N/A"
            normalized["AccuracyValue"] = _safe_float(normalized["Accuracy"])
            rows.append(normalized)

    rows.sort(key=lambda item: item["AccuracyValue"] if item["AccuracyValue"] is not None else -1.0, reverse=True)
    return rows


# ---------------------------------------------------------
# Optional QR code creation with graceful fallback.
# ---------------------------------------------------------

def _create_qr_image(github_url: str) -> Optional[Image]:
    """Create QR code image flowable if qrcode library is available."""
    try:
        import qrcode
    except Exception:
        return None

    try:
        qr = qrcode.QRCode(version=1, box_size=5, border=2)
        qr.add_data(github_url)
        qr.make(fit=True)
        qr_img = qr.make_image(fill_color="black", back_color="white")

        buffer = io.BytesIO()
        qr_img.save(buffer, format="PNG")
        buffer.seek(0)

        image = Image(buffer, width=30 * mm, height=30 * mm)
        image.hAlign = "LEFT"
        return image
    except Exception:
        return None


# ---------------------------------------------------------
# Final assembly function with mandated signature.
# ---------------------------------------------------------

def create_final_report(
    story: List[Any],
    styles: StyleSheet1,
    status: str,
    risk_level: str,
    risk_score: int,
    generated_at: str,
    model_name: str,
    best_accuracy: str,
) -> None:
    """Create final executive summary page with signature and QR section."""
    story.append(_section_band("Executive Banking Final Summary", styles))
    story.append(Spacer(1, 4 * mm))

    final_table = Table(
        [
            ["Best Model Name", model_name],
            ["Accuracy", best_accuracy],
            ["Risk Level", risk_level],
            ["Financial Score", f"{risk_score} / 100"],
            ["Final Decision", status],
        ],
        colWidths=[65 * mm, 105 * mm],
        hAlign="LEFT",
    )
    final_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EFF6FF")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1E3A8A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(final_table)
    story.append(Spacer(1, 8 * mm))

    # QR section (safe fallback if qrcode is unavailable).
    story.append(Paragraph("Scan to view project source code", styles["BodyTextReport"]))
    qr_img = _create_qr_image("https://github.com/your-username/your-fintech-loan-project")
    if qr_img is not None:
        story.append(Spacer(1, 2 * mm))
        story.append(qr_img)
    else:
        story.append(Paragraph("QR module unavailable. QR section skipped safely.", styles["BodyTextMuted"]))

    story.append(Spacer(1, 10 * mm))

    # Digital signature section.
    story.append(Paragraph("Approved by AI Credit Intelligence System", styles["BodyTextReport"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Signature: _______________________________", styles["BodyTextReport"]))
    story.append(Spacer(1, 1.5 * mm))
    story.append(Paragraph(f"Timestamp: {generated_at}", styles["BodyTextMuted"]))
    story.append(Paragraph("System Version: v2.0", styles["BodyTextMuted"]))


def generate_pdf_report(
    applicant_data: Any,
    prediction: Any,
    probability: Any,
    risk_level: Any,
    model_name: Any,
    output_path: Any,
) -> str:
    """Generate premium multi-page fintech PDF report and return output file path."""
    # Normalize and prepare core values.
    applicant = _to_dict(applicant_data)
    status = _normalize_status(prediction)
    prob = _normalize_probability(probability)
    risk_level_text = str(risk_level).strip().title()
    model_name_text = str(model_name)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    risk_score = round((1 - prob) * 100)

    # Prepare output paths and document shell.
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    outputs_dir = output.parent

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="GLOBAL AI FINTECH BANK - Loan Risk Intelligence Report",
        author="GLOBAL AI FINTECH BANK",
    )

    styles = _build_styles()
    story: List[Any] = []

    # 1) Cover page.
    create_cover_page(story, styles, generated_at, model_name_text, applicant)

    # 2) Executive summary page (new requirement).
    create_summary_page(story, styles, status, prob, risk_score)

    # 3) Applicant profile page.
    create_applicant_table(story, styles, applicant)

    # 4) AI decision report page with colored status box and risk gauge text.
    story.append(_section_band("AI Decision Report", styles))
    story.append(Spacer(1, 4 * mm))

    status_box = Table([[Paragraph(f"LOAN STATUS: {status}", styles["StatusText"])]], colWidths=[170 * mm])
    color = _status_color(status, risk_level_text)
    status_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.8, color),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(status_box)
    story.append(Spacer(1, 5 * mm))
    story.append(
        _build_decision_table(
            status=status,
            probability=prob,
            risk_score=risk_score,
            risk_level=risk_level_text,
            model_name=model_name_text,
            prediction_date=generated_at,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(f"Risk Score: {_text_gauge(risk_score)}", styles["BodyTextReport"]))
    story.append(Paragraph("0-30 Low Risk | 31-60 Medium Risk | 61-100 High Risk", styles["BodyTextMuted"]))
    story.append(PageBreak())

    # 5) AI insight page (new requirement).
    story.append(_section_band("AI Insight Section", styles))
    story.append(Spacer(1, 4 * mm))
    for insight in _ai_insights(status):
        story.append(Paragraph(f"- {insight}", styles["BodyTextReport"]))
        story.append(Spacer(1, 2 * mm))
    story.append(PageBreak())

    # 6) Advanced visual section.
    create_visual_section(story, styles, outputs_dir)

    # 7) Enhanced model comparison with ranking and best row highlight.
    story.append(_section_band("Enhanced Model Comparison", styles))
    story.append(Spacer(1, 4 * mm))

    rows = _read_model_comparison(outputs_dir / "model_comparison.csv")
    best_model = model_name_text
    best_accuracy = "N/A"

    if rows:
        table_data = [["Rank", "Model", "Accuracy", "Precision", "Recall", "F1 Score"]]
        for rank, row in enumerate(rows, start=1):
            table_data.append(
                [
                    str(rank),
                    str(row.get("Model", "N/A")),
                    str(row.get("Accuracy", "N/A")),
                    str(row.get("Precision", "N/A")),
                    str(row.get("Recall", "N/A")),
                    str(row.get("F1 Score", "N/A")),
                ]
            )

        best_model = str(rows[0].get("Model", model_name_text))
        best_acc_val = rows[0].get("AccuracyValue")
        best_accuracy = f"{best_acc_val:.4f}" if isinstance(best_acc_val, float) else str(rows[0].get("Accuracy", "N/A"))

        comparison_table = Table(
            table_data,
            colWidths=[16 * mm, 42 * mm, 23 * mm, 23 * mm, 23 * mm, 23 * mm],
            hAlign="LEFT",
        )

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1E3A8A")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BFDBFE")),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]

        for idx in range(1, len(table_data)):
            style_cmds.append(("BACKGROUND", (0, idx), (-1, idx), colors.HexColor("#F8FAFC") if idx % 2 else colors.white))

        # Emphasize best performance row in green.
        style_cmds.extend(
            [
                ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#DCFCE7")),
                ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#166534")),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ]
        )

        comparison_table.setStyle(TableStyle(style_cmds))
        story.append(comparison_table)
    else:
        story.append(Paragraph("Model comparison data not available.", styles["BodyTextMuted"]))

    story.append(PageBreak())

    # 8) Final executive report page with QR + signature.
    create_final_report(
        story=story,
        styles=styles,
        status=status,
        risk_level=risk_level_text,
        risk_score=risk_score,
        generated_at=generated_at,
        model_name=best_model,
        best_accuracy=best_accuracy,
    )

    # Build output and return path.
    document.build(story, onFirstPage=_draw_page_number, onLaterPages=_draw_page_number)
    return str(output)
