
import joblib
import pandas as pd


# ============================================================
# LOAD SAVED MODEL
# ============================================================

MODEL_PATH = "models/merchant_risk_xgboost.pkl"

model_package = joblib.load(MODEL_PATH)

pipeline = model_package["pipeline"]
HIGH_RISK_THRESHOLD = model_package["high_risk_threshold"]


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_merchant_risk(merchant_data):

    merchant_df = pd.DataFrame([merchant_data])

    probabilities = pipeline.predict_proba(
        merchant_df
    )[0]

    low_probability = probabilities[0]
    medium_probability = probabilities[1]
    high_probability = probabilities[2]

    # Apply the tuned High-risk threshold
    if high_probability >= HIGH_RISK_THRESHOLD:

        risk_label = "High"

    elif low_probability >= medium_probability:

        risk_label = "Low"

    else:

        risk_label = "Medium"

    return (
        risk_label,
        low_probability,
        medium_probability,
        high_probability,
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_number(prompt, min_value=None, max_value=None):

    while True:

        try:

            value = float(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue

            return value

        except ValueError:

            print("Please enter a valid number.")


def get_integer(prompt, min_value=None, max_value=None):

    while True:

        try:

            value = int(input(prompt))

            if min_value is not None and value < min_value:
                print(f"Please enter a value >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter a value <= {max_value}.")
                continue

            return value

        except ValueError:

            print("Please enter a whole number.")


def get_choice(prompt, choices):

    print()

    for index, choice in enumerate(choices, start=1):
        print(f"{index}. {choice}")

    while True:

        try:

            selection = int(input(prompt))

            if 1 <= selection <= len(choices):

                return choices[selection - 1]

            print(
                f"Please enter a number from 1 to {len(choices)}."
            )

        except ValueError:

            print("Please enter a valid number.")


def get_binary(prompt):

    while True:

        value = input(prompt).strip()

        if value in ["0", "1"]:

            return int(value)

        print("Please enter 1 for Yes or 0 for No.")


# ============================================================
# COLLECT MERCHANT INFORMATION
# ============================================================

def collect_merchant_data():

    print("\n" + "=" * 60)
    print("MERCHANT RISK SCORING SYSTEM")
    print("=" * 60)

    print(
        "\nEnter the merchant information below."
    )

    print(
        "\n--- BUSINESS INFORMATION ---"
    )

    business_category = get_choice(
        "Select business category:",
        [
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
        ],
    )

    business_age_months = get_integer(
        "\nBusiness age (months): ",
        0,
    )

    business_type = get_choice(
        "Select business type:",
        [
            "Private Limited",
            "LLP",
            "Partnership",
            "Sole Proprietorship",
        ],
    )

    kyc_verification_level = get_choice(
        "Select KYC verification level:",
        [
            "Basic",
            "Enhanced",
            "Full",
        ],
    )

    registered_state = get_choice(
        "Select registered state:",
        [
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
        ],
    )

    website_verified = get_binary(
        "\nWebsite verified? (1 = Yes, 0 = No): "
    )

    print(
        "\n--- TRANSACTION INFORMATION ---"
    )

    avg_monthly_txn_volume = get_number(
        "Average monthly transaction volume: ",
        0,
    )

    avg_transaction_value = get_number(
        "Average transaction value: ",
        0,
    )

    txn_growth_rate_30d = get_number(
        "Transaction growth rate (30 days, %): "
    )

    txn_success_rate = get_number(
        "Transaction success rate (%): ",
        0,
        100,
    )

    refund_ratio = get_number(
        "Refund ratio (%): ",
        0,
        100,
    )

    chargeback_ratio = get_number(
        "Chargeback ratio (%): ",
        0,
        100,
    )

    night_time_txn_percentage = get_number(
        "Night-time transaction percentage (%): ",
        0,
        100,
    )

    print(
        "\n--- SETTLEMENT & BANKING ---"
    )

    settlement_delay_requests = get_integer(
        "Settlement delay requests: ",
        0,
    )

    bank_account_age_days = get_integer(
        "Bank account age (days): ",
        0,
    )

    reserve_balance_ratio = get_number(
        "Reserve balance ratio (%): ",
        0,
        100,
    )

    multiple_bank_accounts_flag = get_binary(
        "Multiple bank accounts? (1 = Yes, 0 = No): "
    )

    print(
        "\n--- COMPLIANCE & RISK ---"
    )

    previous_suspension_flag = get_binary(
        "Previous suspension? (1 = Yes, 0 = No): "
    )

    complaint_count_90d = get_integer(
        "Complaint count in last 90 days: ",
        0,
    )

    blacklist_match_score = get_number(
        "Blacklist match score: ",
        0,
        100,
    )

    return {
        "business_category": business_category,
        "business_age_months": business_age_months,
        "business_type": business_type,
        "kyc_verification_level": kyc_verification_level,
        "registered_state": registered_state,
        "website_verified": website_verified,
        "avg_monthly_txn_volume": avg_monthly_txn_volume,
        "avg_transaction_value": avg_transaction_value,
        "txn_growth_rate_30d": txn_growth_rate_30d,
        "txn_success_rate": txn_success_rate,
        "refund_ratio": refund_ratio,
        "chargeback_ratio": chargeback_ratio,
        "night_time_txn_percentage": night_time_txn_percentage,
        "settlement_delay_requests": settlement_delay_requests,
        "bank_account_age_days": bank_account_age_days,
        "reserve_balance_ratio": reserve_balance_ratio,
        "multiple_bank_accounts_flag": multiple_bank_accounts_flag,
        "previous_suspension_flag": previous_suspension_flag,
        "complaint_count_90d": complaint_count_90d,
        "blacklist_match_score": blacklist_match_score,
    }


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    risk_label,
    low_probability,
    medium_probability,
    high_probability,
):

    print("\n" + "=" * 60)
    print("RISK ASSESSMENT RESULT")
    print("=" * 60)

    print(
        f"\nRisk Level: {risk_label.upper()}"
    )

    print("\nRisk probabilities:")

    print(
        f"Low:       {low_probability * 100:.2f}%"
    )

    print(
        f"Medium:    {medium_probability * 100:.2f}%"
    )

    print(
        f"High:      {high_probability * 100:.2f}%"
    )

    print(
        f"\nHigh-risk threshold: "
        f"{HIGH_RISK_THRESHOLD * 100:.0f}%"
    )

    print("\nRecommendation:")

    if risk_label == "High":

        print(
            "MANUAL REVIEW REQUIRED"
        )

    elif risk_label == "Medium":

        print(
            "ENHANCED MONITORING RECOMMENDED"
        )

    else:

        print(
            "STANDARD MONITORING"
        )

    print("\n" + "=" * 60)


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    merchant_data = collect_merchant_data()

    (
        risk_label,
        low_probability,
        medium_probability,
        high_probability,
    ) = predict_merchant_risk(
        merchant_data
    )

    display_result(
        risk_label,
        low_probability,
        medium_probability,
        high_probability,
    )
