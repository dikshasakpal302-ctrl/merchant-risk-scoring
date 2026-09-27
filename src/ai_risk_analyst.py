"""
AI Risk Analyst
---------------
AI-powered investigation and decision-support layer.

IMPORTANT:
- This module does NOT change the XGBoost prediction.
- XGBoost remains the source of the risk classification.
- OpenAI is used only to interpret the existing ML result.
- A deterministic local fallback is provided if OpenAI is unavailable.
"""

import os
from openai import OpenAI


# ============================================================
# LOCAL RISK SIGNAL DETECTION
# ============================================================

def detect_risk_signals(merchant_data: dict) -> list:
    """
    Detect potentially important risk signals from merchant data.

    These rules are NOT used to classify the merchant.
    They are only used to provide context to the AI analyst
    and as a fallback when OpenAI is unavailable.
    """

    signals = []

    if merchant_data.get("previous_suspension_flag", 0) == 1:
        signals.append(
            "Previous suspension detected."
        )

    if merchant_data.get("chargeback_ratio", 0) >= 3:
        signals.append(
            f"Elevated chargeback ratio "
            f"({merchant_data['chargeback_ratio']:.1f}%)."
        )

    if merchant_data.get("refund_ratio", 0) >= 7:
        signals.append(
            f"Elevated refund ratio "
            f"({merchant_data['refund_ratio']:.1f}%)."
        )

    if merchant_data.get("kyc_verification_level") == "Basic":
        signals.append(
            "Basic KYC verification level."
        )

    if merchant_data.get("website_verified", 1) == 0:
        signals.append(
            "Website verification is incomplete."
        )

    if merchant_data.get("multiple_bank_accounts_flag", 0) == 1:
        signals.append(
            "Multiple linked bank accounts."
        )

    if merchant_data.get("complaint_count_90d", 0) >= 10:
        signals.append(
            f"High complaint volume "
            f"({merchant_data['complaint_count_90d']} complaints in 90 days)."
        )

    if merchant_data.get("blacklist_match_score", 0) >= 20:
        signals.append(
            f"Elevated blacklist match score "
            f"({merchant_data['blacklist_match_score']:.1f})."
        )

    if merchant_data.get("txn_growth_rate_30d", 0) >= 50:
        signals.append(
            f"Rapid transaction growth "
            f"({merchant_data['txn_growth_rate_30d']:.1f}% over 30 days)."
        )

    if merchant_data.get("txn_success_rate", 100) < 85:
        signals.append(
            f"Low transaction success rate "
            f"({merchant_data['txn_success_rate']:.1f}%)."
        )

    if merchant_data.get("night_time_txn_percentage", 0) >= 40:
        signals.append(
            f"High night-time transaction activity "
            f"({merchant_data['night_time_txn_percentage']:.1f}%)."
        )

    if merchant_data.get("bank_account_age_days", 99999) < 180:
        signals.append(
            f"Relatively new bank account "
            f"({merchant_data['bank_account_age_days']} days old)."
        )

    if merchant_data.get("settlement_delay_requests", 0) >= 2:
        signals.append(
            f"Repeated settlement delay requests "
            f"({merchant_data['settlement_delay_requests']})."
        )

    if merchant_data.get("reserve_balance_ratio", 100) < 5:
        signals.append(
            f"Low reserve balance ratio "
            f"({merchant_data['reserve_balance_ratio']:.1f}%)."
        )

    return signals


# ============================================================
# CONTEXTUAL SIGNALS
# ============================================================

def detect_contextual_signals(merchant_data: dict) -> list:
    """
    Identify signals that may require context but should not
    automatically be treated as suspicious.
    """

    signals = []

    if merchant_data.get("txn_success_rate", 100) >= 85:
        signals.append(
            f"Transaction success rate is "
            f"{merchant_data['txn_success_rate']:.1f}%."
        )

    if merchant_data.get("night_time_txn_percentage", 0) > 0:
        signals.append(
            f"{merchant_data['night_time_txn_percentage']:.1f}% "
            "of transactions occur at night."
        )

    if merchant_data.get("txn_growth_rate_30d", 0) > 0:
        signals.append(
            f"Transaction activity changed by "
            f"{merchant_data['txn_growth_rate_30d']:.1f}% "
            "over the last 30 days."
        )

    if merchant_data.get("business_age_months", 0) > 0:
        signals.append(
            f"Business age is "
            f"{merchant_data['business_age_months']} months."
        )

    if merchant_data.get("bank_account_age_days", 0) > 0:
        signals.append(
            f"Linked bank account is "
            f"{merchant_data['bank_account_age_days']} days old."
        )

    if merchant_data.get("website_verified", 0) == 1:
        signals.append(
            "Website is verified."
        )

    if merchant_data.get("previous_suspension_flag", 0) == 0:
        signals.append(
            "No previous suspension is recorded."
        )

    if merchant_data.get("multiple_bank_accounts_flag", 0) == 0:
        signals.append(
            "No multiple-bank-account flag is recorded."
        )

    return signals


# ============================================================
# LOCAL AI RISK ANALYST
# ============================================================

def generate_local_risk_analysis(
    merchant_data: dict,
    risk_label: str,
    low_probability: float,
    medium_probability: float,
    high_probability: float,
    high_risk_threshold: float,
) -> dict:
    """
    Deterministic fallback analyst.

    This does not make or modify the ML decision.
    """

    risk_signals = detect_risk_signals(
        merchant_data
    )

    contextual_signals = detect_contextual_signals(
        merchant_data
    )

    # --------------------------------------------------------
    # Risk summary
    # --------------------------------------------------------

    summary = (
        f"The XGBoost model classifies this merchant as "
        f"{risk_label} Risk with a "
        f"{high_probability * 100:.2f}% High-risk probability. "
        f"The configured High-risk threshold is "
        f"{high_risk_threshold * 100:.0f}%. "
        f"The AI Risk Analyst does not modify this classification."
    )

    # --------------------------------------------------------
    # False-positive considerations
    # --------------------------------------------------------

    false_positive_points = []

    if merchant_data.get("blacklist_match_score", 0) < 20:
        false_positive_points.append(
            "The blacklist match score is relatively low and "
            "should be validated before treating it as a strong signal."
        )

    if merchant_data.get("complaint_count_90d", 0) < 10:
        false_positive_points.append(
            "The complaint count is not independently high, "
            "so complaint context and outcomes should be reviewed."
        )

    if merchant_data.get("txn_growth_rate_30d", 0) < 50:
        false_positive_points.append(
            "Transaction growth is not extreme and may reflect "
            "normal business expansion."
        )

    if merchant_data.get("night_time_txn_percentage", 0) < 40:
        false_positive_points.append(
            "Night-time transaction activity is below the "
            "local analyst's elevated-activity threshold and "
            "may be normal for the merchant's business model."
        )

    if merchant_data.get("previous_suspension_flag", 0) == 0:
        false_positive_points.append(
            "No previous suspension is recorded in the supplied data."
        )

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    recommendations = []

    if merchant_data.get("kyc_verification_level") == "Basic":
        recommendations.append(
            "Complete enhanced KYC and verify business ownership "
            "and authorized representatives."
        )

    if merchant_data.get("blacklist_match_score", 0) >= 20:
        recommendations.append(
            "Investigate the blacklist match source and determine "
            "whether the match is valid or a false positive."
        )

    if merchant_data.get("chargeback_ratio", 0) >= 3:
        recommendations.append(
            "Review recent chargeback transactions and identify "
            "common dispute patterns."
        )

    if merchant_data.get("refund_ratio", 0) >= 7:
        recommendations.append(
            "Review refund history and determine the primary "
            "causes of refunds."
        )

    if merchant_data.get("complaint_count_90d", 0) >= 10:
        recommendations.append(
            "Review customer complaints and identify recurring "
            "issues or behavioral patterns."
        )

    if merchant_data.get("multiple_bank_accounts_flag", 0) == 1:
        recommendations.append(
            "Validate all linked settlement bank accounts against "
            "the verified merchant entity."
        )

    if merchant_data.get("previous_suspension_flag", 0) == 1:
        recommendations.append(
            "Review the reason for the previous suspension and "
            "whether the underlying issue has been resolved."
        )

    if merchant_data.get("txn_growth_rate_30d", 0) >= 50:
        recommendations.append(
            "Validate rapid transaction growth against invoices, "
            "orders, settlements, and business activity."
        )

    if not recommendations:
        recommendations.append(
            "Review KYC, transaction history, customer complaints, "
            "settlements, and other available merchant evidence."
        )

    recommendations.append(
        "Keep the final decision with an authorized human reviewer "
        "and apply any restrictions according to internal policy."
    )

    # --------------------------------------------------------
    # Human decision
    # --------------------------------------------------------

    if risk_label == "High":
        decision = "MANUAL REVIEW REQUIRED"
    elif risk_label == "Medium":
        decision = "ENHANCED MONITORING RECOMMENDED"
    else:
        decision = "STANDARD MONITORING"

    # --------------------------------------------------------
    # Structured analysis
    # --------------------------------------------------------

    analysis = f"""
### RISK SUMMARY

The XGBoost model classifies this merchant as **{risk_label} Risk**, with a **{high_probability * 100:.2f}% High-risk probability** against a configured **{high_risk_threshold * 100:.0f}% High-risk threshold**.

This result warrants an appropriate level of review, but the model result does not by itself establish fraud or wrongdoing.

### KEY RISK SIGNALS

"""

    if risk_signals:

        for signal in risk_signals:
            analysis += f"- {signal}\n"

    else:

        analysis += (
            "- No predefined elevated-risk signals were detected "
            "by the local analyst.\n"
        )

    analysis += """

### CONTEXTUAL SIGNALS

"""

    if contextual_signals:

        for signal in contextual_signals:
            analysis += f"- {signal}\n"

    else:

        analysis += (
            "- No additional contextual signals identified.\n"
        )

    analysis += """

### POTENTIAL FALSE-POSITIVE CONSIDERATIONS

"""

    for point in false_positive_points:
        analysis += f"- {point}\n"

    analysis += """

### INVESTIGATION RECOMMENDATIONS

"""

    for index, recommendation in enumerate(
        recommendations,
        start=1,
    ):
        analysis += (
            f"{index}. {recommendation}\n"
        )

    analysis += """

### HUMAN REVIEW DECISION

"""

    analysis += (
        f"**{decision}**\n\n"
        "The final decision should be made by an authorized "
        "human reviewer after evaluating the available evidence."
    )

    return {
        "source": "local",
        "analysis": analysis,
        "signals": risk_signals,
        "recommendations": recommendations,
        "decision": decision,
    }


# ============================================================
# OPENAI ANALYST
# ============================================================

def generate_openai_risk_analysis(
    merchant_data: dict,
    risk_label: str,
    low_probability: float,
    medium_probability: float,
    high_probability: float,
    high_risk_threshold: float,
) -> dict:
    """
    Generate an AI investigation analysis using OpenAI.

    The model receives the existing XGBoost output and merchant
    features. It is explicitly instructed not to modify the
    classification.
    """

    api_key = os.getenv(
        "OPENAI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured."
        )

    client = OpenAI(
        api_key=api_key
    )

    system_prompt = """
You are an AI Risk Analyst assisting a payment-risk investigator.

Your role is decision support, not autonomous decision-making.

IMPORTANT RULES:

1. The XGBoost model is the source of the merchant risk classification.
2. NEVER change, recalculate, override, or reinterpret the ML classification.
3. NEVER claim that a merchant is fraudulent.
4. Do not invent facts that are not present in the supplied data.
5. Distinguish genuine risk signals from contextual signals.
6. Identify potential false-positive explanations where appropriate.
7. Recommend specific evidence an investigator should review.
8. The final decision must remain with a human reviewer.
9. Be concise, factual, and suitable for a fintech risk investigation workflow.

Return exactly these sections:

RISK SUMMARY

KEY RISK SIGNALS

CONTEXTUAL SIGNALS

POTENTIAL FALSE-POSITIVE CONSIDERATIONS

INVESTIGATION RECOMMENDATIONS

ADDITIONAL EVIDENCE TO REQUEST

HUMAN REVIEW DECISION
"""

    user_prompt = f"""
Existing XGBoost assessment:

Risk classification:
{risk_label}

Low probability:
{low_probability * 100:.2f}%

Medium probability:
{medium_probability * 100:.2f}%

High probability:
{high_probability * 100:.2f}%

High-risk threshold:
{high_risk_threshold * 100:.0f}%

Merchant data:

{merchant_data}

Analyze this merchant as an investigator's decision-support assistant.

Focus on:

- which supplied signals actually deserve attention
- which signals require business context
- possible false-positive explanations
- what evidence should be requested
- what an investigator should verify next

Do not change the XGBoost classification.
Do not accuse the merchant of fraud.
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        instructions=system_prompt,
        input=user_prompt,
    )

    analysis = response.output_text

    return {
        "source": "openai",
        "analysis": analysis,
        "signals": detect_risk_signals(
            merchant_data
        ),
        "recommendations": [],
        "decision": (
            "MANUAL REVIEW REQUIRED"
            if risk_label == "High"
            else (
                "ENHANCED MONITORING RECOMMENDED"
                if risk_label == "Medium"
                else "STANDARD MONITORING"
            )
        ),
    }


# ============================================================
# MAIN ANALYST FUNCTION
# ============================================================

def analyze_merchant_risk(
    merchant_data: dict,
    risk_label: str,
    low_probability: float,
    medium_probability: float,
    high_probability: float,
    high_risk_threshold: float,
) -> dict:
    """
    Main entry point.

    Tries OpenAI first.

    If OpenAI is unavailable, automatically falls back to the
    deterministic local Risk Analyst.
    """

    try:

        return generate_openai_risk_analysis(
            merchant_data=merchant_data,
            risk_label=risk_label,
            low_probability=low_probability,
            medium_probability=medium_probability,
            high_probability=high_probability,
            high_risk_threshold=high_risk_threshold,
        )

    except Exception:

        return generate_local_risk_analysis(
            merchant_data=merchant_data,
            risk_label=risk_label,
            low_probability=low_probability,
            medium_probability=medium_probability,
            high_probability=high_probability,
            high_risk_threshold=high_risk_threshold,
        )