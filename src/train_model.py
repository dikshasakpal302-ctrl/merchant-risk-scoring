# ============================================================
# MERCHANT RISK CLASSIFICATION - XGBOOST
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("MERCHANT RISK CLASSIFICATION - XGBOOST")
print("=" * 60)

data_path = "data/merchant_risk_dataset.csv"

df = pd.read_csv(data_path)

print(f"\nDataset shape: {df.shape}")


# ============================================================
# 2. REMOVE ID AND TARGET LEAKAGE
# ============================================================

X = df.drop(
    columns=[
        "merchant_id",
        "risk_score",
        "risk_label",
    ]
)

y = df["risk_label"]


print("\nRemoved:")
print("- merchant_id")
print("- risk_score (target leakage)")


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    y.value_counts()
)


# ============================================================
# 4. DEFINE FEATURES
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


print("\nCategorical features:")
print(categorical_features)

print("\nNumeric features:")
print(numeric_features)


# ============================================================
# 5. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First split:
# 80% training
# 20% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)


# Second split:
# 10% validation
# 10% testing

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)


print(f"\nTraining samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================
# 6. ENCODE TARGET
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
y_test_encoded = y_test.map(label_mapping)


# ============================================================
# 7. CALCULATE CLASS WEIGHTS
# ============================================================

class_counts = (
    y_train_encoded
    .value_counts()
    .sort_index()
)


total_samples = len(
    y_train_encoded
)

num_classes = len(
    class_counts
)


class_weights = {
    cls: total_samples /
    (num_classes * count)
    for cls, count
    in class_counts.items()
}


sample_weights = (
    y_train_encoded
    .map(class_weights)
    .values
)


print("\nClass counts in training data:")

print(class_counts)


print("\nClass weights:")

for cls, weight in class_weights.items():

    print(
        f"{reverse_mapping[cls]}: "
        f"{weight:.3f}"
    )


# ============================================================
# 8. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
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
# 9. XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(

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


# ============================================================
# 10. CREATE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            xgb_model,
        ),
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)


pipeline.fit(
    X_train,
    y_train_encoded,
    model__sample_weight=sample_weights,
)


print("\nTraining completed!")


# ============================================================
# 12. VALIDATION PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 60)


y_val_encoded = y_val.map(
    label_mapping
)


val_probabilities = (
    pipeline.predict_proba(X_val)
)


val_high_risk_probability = (
    val_probabilities[:, 2]
)


actual_val_high = (
    y_val_encoded == 2
).values


print("\nValidation predictions generated!")


# ============================================================
# 13. HIGH-RISK THRESHOLD SELECTION
# ============================================================

print("\n" + "=" * 60)
print("HIGH-RISK THRESHOLD SELECTION")
print("=" * 60)


thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


threshold_results = []


for threshold in thresholds:

    predicted_high = (
        val_high_risk_probability
        >= threshold
    )

    true_positive = (
        predicted_high
        & actual_val_high
    ).sum()

    false_positive = (
        predicted_high
        & ~actual_val_high
    ).sum()

    false_negative = (
        ~predicted_high
        & actual_val_high
    ).sum()

    precision = (
        true_positive /
        (true_positive + false_positive)
        if (
            true_positive
            + false_positive
        ) > 0
        else 0
    )

    recall = (
        true_positive /
        (true_positive + false_negative)
        if (
            true_positive
            + false_negative
        ) > 0
        else 0
    )

    threshold_results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "true_positives": true_positive,

        "false_positives": false_positive,

        "false_negatives": false_negative,

    })


threshold_df = pd.DataFrame(
    threshold_results
)


print("\nValidation threshold comparison:")

print(
    threshold_df.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "precision": "{:.3f}".format,
            "recall": "{:.3f}".format,
        },
    )
)


# ============================================================
# 14. SELECT BEST THRESHOLD
# ============================================================

# Business objective:
# prioritize detection of High-risk merchants.
#
# Requirement:
# High-risk recall >= 85%
#
# Among thresholds meeting that requirement,
# choose the threshold with the highest precision.

eligible_thresholds = (
    threshold_df[
        threshold_df["recall"] >= 0.85
    ]
)


if not eligible_thresholds.empty:

    best_row = (
        eligible_thresholds.loc[
            eligible_thresholds[
                "precision"
            ].idxmax()
        ]
    )

else:

    best_row = (
        threshold_df.loc[
            threshold_df[
                "recall"
            ].idxmax()
        ]
    )


HIGH_RISK_THRESHOLD = float(
    best_row["threshold"]
)


print("\n" + "=" * 60)
print("SELECTED HIGH-RISK THRESHOLD")
print("=" * 60)


print(
    f"\nSelected threshold: "
    f"{HIGH_RISK_THRESHOLD:.2f}"
)

print(
    f"Validation precision: "
    f"{best_row['precision']:.3f}"
)

print(
    f"Validation recall: "
    f"{best_row['recall']:.3f}"
)

print(
    f"Validation true positives: "
    f"{int(best_row['true_positives'])}"
)

print(
    f"Validation false positives: "
    f"{int(best_row['false_positives'])}"
)

print(
    f"Validation false negatives: "
    f"{int(best_row['false_negatives'])}"
)


# ============================================================
# 15. FINAL TEST SET EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST SET EVALUATION")
print("=" * 60)


# Standard XGBoost predictions

y_pred_encoded = (
    pipeline.predict(X_test)
)


y_pred = pd.Series(
    y_pred_encoded
).map(reverse_mapping)


# Test probabilities

test_probabilities = (
    pipeline.predict_proba(X_test)
)


test_high_risk_probability = (
    test_probabilities[:, 2]
)


# ============================================================
# 16. TEST MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)


print(
    f"\nTest Accuracy: "
    f"{accuracy:.4f}"
)


print("\nTest Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[
            "High",
            "Low",
            "Medium",
        ],
        zero_division=0,
    )
)


# ============================================================
# 17. TEST CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "Low",
        "Medium",
        "High",
    ],
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


print("\nFinal Test Confusion Matrix:")

print(cm_df)


# ============================================================
# 18. FINAL HIGH-RISK THRESHOLD PERFORMANCE
# ============================================================

actual_test_high = (
    y_test_encoded == 2
).values


threshold_high_predictions = (
    test_high_risk_probability
    >= HIGH_RISK_THRESHOLD
)


true_positive = (
    threshold_high_predictions
    & actual_test_high
).sum()


false_positive = (
    threshold_high_predictions
    & ~actual_test_high
).sum()


false_negative = (
    ~threshold_high_predictions
    & actual_test_high
).sum()


high_precision = (
    true_positive /
    (true_positive + false_positive)
    if (
        true_positive
        + false_positive
    ) > 0
    else 0
)


high_recall = (
    true_positive /
    (true_positive + false_negative)
    if (
        true_positive
        + false_negative
    ) > 0
    else 0
)


print("\n" + "=" * 60)
print("FINAL HIGH-RISK THRESHOLD PERFORMANCE")
print("=" * 60)


print(
    f"\nThreshold: "
    f"{HIGH_RISK_THRESHOLD:.2f}"
)

print(
    f"High-risk precision: "
    f"{high_precision:.3f}"
)

print(
    f"High-risk recall: "
    f"{high_recall:.3f}"
)

print(
    f"True positives: "
    f"{true_positive}"
)

print(
    f"False positives: "
    f"{false_positive}"
)

print(
    f"False negatives: "
    f"{false_negative}"
)


# ============================================================
# 19. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("TOP 15 FEATURE IMPORTANCES")
print("=" * 60)


fitted_preprocessor = (
    pipeline.named_steps[
        "preprocessor"
    ]
)


fitted_model = (
    pipeline.named_steps[
        "model"
    ]
)


feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)


feature_importance_df = pd.DataFrame({

    "feature": feature_names,

    "importance":
        fitted_model.feature_importances_,

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        "importance",
        ascending=False,
    )
)


print(
    feature_importance_df
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 20. SAVE FEATURE IMPORTANCE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)


feature_importance_df.to_csv(
    "outputs/feature_importance.csv",
    index=False,
)


# ============================================================
# 21. SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)


model_package = {

    "pipeline": pipeline,

    "high_risk_threshold":
        HIGH_RISK_THRESHOLD,

    "label_mapping":
        label_mapping,

    "reverse_mapping":
        reverse_mapping,

}


model_path = (
    "models/merchant_risk_xgboost.pkl"
)


joblib.dump(
    model_package,
    model_path,
)


print("\n" + "=" * 60)
print("FINAL MODEL SAVED")
print("=" * 60)


print(
    f"Location: {model_path}"
)

print(
    f"High-risk threshold: "
    f"{HIGH_RISK_THRESHOLD}"
)


print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)