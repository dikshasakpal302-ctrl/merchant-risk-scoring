import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. SETTINGS
# ==========================================

DATA_PATH = "data/merchant_risk_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("MERCHANT RISK SCORING - EDA")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ==========================================
# 3. RISK LABEL DISTRIBUTION
# ==========================================

print("\nRisk label counts:")
print(df["risk_label"].value_counts())

print("\nRisk label percentages:")
print(
    df["risk_label"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="risk_label",
    order=["Low", "Medium", "High"]
)

plt.title("Merchant Risk Label Distribution")
plt.xlabel("Risk Label")
plt.ylabel("Number of Merchants")
plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_risk_label_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 4. RISK BY BUSINESS CATEGORY
# ==========================================

risk_by_category = pd.crosstab(
    df["business_category"],
    df["risk_label"],
    normalize="index"
) * 100

print("\nRisk percentage by business category:")
print(risk_by_category.round(2))


risk_by_category.plot(
    kind="bar",
    stacked=True,
    figsize=(12, 6)
)

plt.title("Risk Distribution by Business Category")
plt.xlabel("Business Category")
plt.ylabel("Percentage")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Risk Label")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/02_risk_by_category.png",
    dpi=300
)

plt.close()


# ==========================================
# 5. NUMERIC CORRELATION
# ==========================================

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap - Numeric Features")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/03_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# 6. FEATURES CORRELATED WITH RISK SCORE
# ==========================================

risk_correlations = (
    numeric_df
    .corr()["risk_score"]
    .drop("risk_score")
    .sort_values(ascending=False)
)

print("\nFeatures positively correlated with risk_score:")
print(risk_correlations)


plt.figure(figsize=(9, 6))

risk_correlations.sort_values().plot(kind="barh")

plt.title("Features Correlated with Risk Score")
plt.xlabel("Correlation with Risk Score")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/04_risk_score_correlations.png",
    dpi=300
)

plt.close()


# ==========================================
# 7. CHARGEBACK RATIO BY RISK
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_label",
    y="chargeback_ratio",
    order=["Low", "Medium", "High"]
)

plt.title("Chargeback Ratio by Risk Level")
plt.xlabel("Risk Label")
plt.ylabel("Chargeback Ratio")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/05_chargeback_by_risk.png",
    dpi=300
)

plt.close()


# ==========================================
# 8. REFUND RATIO BY RISK
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="risk_label",
    y="refund_ratio",
    order=["Low", "Medium", "High"]
)

plt.title("Refund Ratio by Risk Level")
plt.xlabel("Risk Label")
plt.ylabel("Refund Ratio")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/06_refund_by_risk.png",
    dpi=300
)

plt.close()


# ==========================================
# 9. RISK SCORE DISTRIBUTION
# ==========================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="risk_score",
    hue="risk_label",
    bins=30,
    kde=True
)

plt.title("Risk Score Distribution")
plt.xlabel("Risk Score")
plt.ylabel("Number of Merchants")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/07_risk_score_distribution.png",
    dpi=300
)

plt.close()


# ==========================================
# 10. COMPLETION
# ==========================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\nAll plots have been saved to: {OUTPUT_DIR}/")