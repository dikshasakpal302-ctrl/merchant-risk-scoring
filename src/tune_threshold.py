
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from xgboost import XGBClassifier


print("=" * 60)
print("MERCHANT RISK - HIGH-RISK THRESHOLD TUNING")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/merchant_risk_dataset.csv")

# Remove ID and target leakage
df = df.drop(columns=["merchant_id", "risk_score"])

X = df.drop(columns=["risk_label"])
y = df["risk_label"]

# ============================================================
# 2. FEATURES
# ============================================================

categorical_features = [
    "business_category",
    "business_type",
    "kyc_verification_level",
    "registered_state",
]

numeric_features = [
    "business_age_months",
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
# 3. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)

# ============================================================
# 5. TARGET ENCODING
# ============================================================

label_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}

reverse_mapping = {
    0: "Low",
    1: "Medium",
    2: "High",
}

y_train_encoded = y_train.map(label_mapping)

# ============================================================
# 6. CLASS WEIGHTS
# ============================================================

class_counts = y_train_encoded.value_counts().sort_index()

total_samples = len(y_train_encoded)
num_classes = len(class_counts)

class_weights = {
    cls: total_samples / (num_classes * count)
    for cls, count in class_counts.items()
}

sample_weights = y_train_encoded.map(class_weights).values

print("\nClass weights:")
for cls, weight in class_weights.items():
    print(f"{reverse_mapping[cls]}: {weight:.3f}")

# ============================================================
# 7. XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train_encoded,
    model__sample_weight=sample_weights,
)

print("Training completed!")

# ============================================================
# 8. GET CLASS PROBABILITIES
# ============================================================

probabilities = pipeline.predict_proba(X_test)

# Probability columns:
# 0 = Low
# 1 = Medium
# 2 = High

high_probability = probabilities[:, 2]

# ============================================================
# 9. BASELINE
# ============================================================

baseline_prediction = np.argmax(probabilities, axis=1)

baseline_labels = pd.Series(
    baseline_prediction
).map(reverse_mapping)

print("\n" + "=" * 60)
print("BASELINE PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy_score(y_test, baseline_labels):.4f}")

print(
    classification_report(
        y_test,
        baseline_labels,
        labels=["High", "Low", "Medium"],
        zero_division=0,
    )
)

# ============================================================
# 10. THRESHOLD EXPERIMENT
# ============================================================

print("\n" + "=" * 60)
print("HIGH-RISK THRESHOLD EXPERIMENT")
print("=" * 60)

print(
    "\nLowering the High-risk threshold should increase "
    "High-risk recall, but may reduce precision."
)

thresholds = [
    0.50,
    0.45,
    0.40,
    0.35,
    0.30,
    0.25,
    0.20,
]

results = []

for threshold in thresholds:

    predictions = []

    for i in range(len(probabilities)):

        high_prob = probabilities[i, 2]

        if high_prob >= threshold:
            prediction = "High"

        else:
            # If not High, choose between Low and Medium
            if probabilities[i, 0] >= probabilities[i, 1]:
                prediction = "Low"
            else:
                prediction = "Medium"

        predictions.append(prediction)

    predictions = np.array(predictions)

    report = classification_report(
        y_test,
        predictions,
        labels=["High", "Low", "Medium"],
        output_dict=True,
        zero_division=0,
    )

    high_precision = report["High"]["precision"]
    high_recall = report["High"]["recall"]
    high_f1 = report["High"]["f1-score"]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    results.append(
        {
            "threshold": threshold,
            "accuracy": accuracy,
            "high_precision": high_precision,
            "high_recall": high_recall,
            "high_f1": high_f1,
        }
    )

results_df = pd.DataFrame(results)

print(
    results_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "accuracy": "{:.4f}".format,
            "high_precision": "{:.4f}".format,
            "high_recall": "{:.4f}".format,
            "high_f1": "{:.4f}".format,
        },
    )
)

# ============================================================
# 11. BEST HIGH-RISK F1
# ============================================================

best_row = results_df.loc[
    results_df["high_f1"].idxmax()
]

print("\n" + "=" * 60)
print("BEST HIGH-RISK THRESHOLD")
print("=" * 60)

print(
    f"\nThreshold: {best_row['threshold']:.2f}"
)

print(
    f"Accuracy: {best_row['accuracy']:.4f}"
)

print(
    f"High-risk precision: "
    f"{best_row['high_precision']:.4f}"
)

print(
    f"High-risk recall: "
    f"{best_row['high_recall']:.4f}"
)

print(
    f"High-risk F1: "
    f"{best_row['high_f1']:.4f}"
)

# ============================================================
# 12. CONFUSION MATRIX FOR BEST THRESHOLD
# ============================================================

best_threshold = best_row["threshold"]

best_predictions = []

for i in range(len(probabilities)):

    if probabilities[i, 2] >= best_threshold:
        prediction = "High"

    else:
        if probabilities[i, 0] >= probabilities[i, 1]:
            prediction = "Low"
        else:
            prediction = "Medium"

    best_predictions.append(prediction)

best_predictions = np.array(best_predictions)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["Low", "Medium", "High"],
)

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Low",
        "Actual Medium",
        "Actual High",
    ],
    columns=[
        "Predicted Low",
        "Predicted Medium",
        "Predicted High",
    ],
)

print("\nConfusion Matrix:")
print(cm_df)

print("\n" + "=" * 60)
print("THRESHOLD TUNING COMPLETED")
print("=" * 60)
