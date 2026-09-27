# Merchant Risk Scoring System

An end-to-end machine learning project that classifies merchants into **Low, Medium, and High risk** categories using business, transaction, banking, compliance, and behavioral risk indicators.

## Project Overview

Merchant risk assessment is important for payment platforms and financial businesses to identify merchants that may require additional monitoring or manual review.

This project builds a machine learning pipeline that:

* Performs exploratory data analysis (EDA)
* Identifies important merchant risk indicators
* Compares a Random Forest baseline with XGBoost
* Handles class imbalance, especially the relatively rare High-risk class
* Uses a validation set to tune the High-risk decision threshold
* Evaluates the final model on an untouched test set
* Provides an interactive command-line prediction system for new merchants

## Problem Statement

Given merchant-level business and transaction information, predict the merchant's risk category:

* **Low** — standard monitoring
* **Medium** — enhanced monitoring recommended
* **High** — manual review required

The main objective is not only overall classification accuracy, but also improving the detection of **High-risk merchants**, where missing a risky merchant can be more costly than incorrectly flagging a lower-risk merchant.

## Dataset

The dataset contains **5,000 merchant records** and **23 columns**.

Important feature groups include:

### Business Information

* Business category
* Business age
* Business type
* KYC verification level
* Registered state
* Website verification

### Transaction Information

* Average monthly transaction volume
* Average transaction value
* 30-day transaction growth
* Transaction success rate
* Refund ratio
* Chargeback ratio
* Night-time transaction percentage

### Settlement & Banking

* Settlement delay requests
* Bank account age
* Reserve balance ratio
* Multiple bank accounts flag

### Compliance & Risk Indicators

* Previous suspension flag
* Complaint count in the last 90 days
* Blacklist match score

### Target

`risk_label`

Classes:

| Risk Level | Count | Percentage |
| ---------- | ----: | ---------: |
| Low        | 2,298 |     45.96% |
| Medium     | 2,563 |     51.26% |
| High       |   139 |      2.78% |

The High-risk class is significantly smaller than the other classes, making class imbalance an important modeling consideration.

## Data Leakage Prevention

The dataset contains a `risk_score` column in addition to `risk_label`.

Exploratory analysis showed that `risk_score` strongly separates the three risk categories. Using it as a model feature would therefore introduce target leakage and make the model evaluation unrealistic.

For this reason, the final model excludes:

* `merchant_id` — identifier, not a predictive feature
* `risk_score` — excluded to prevent target leakage

The model predicts risk using the underlying merchant characteristics instead.

## Exploratory Data Analysis

EDA was performed to understand:

* Risk-label distribution
* Risk levels across business categories
* Correlations between numerical features and risk score
* Chargeback behavior by risk level
* Refund behavior by risk level
* Risk-score distribution

Key risk indicators identified during analysis included:

* Previous suspension
* Chargeback ratio
* Refund ratio
* Multiple bank accounts
* Settlement delay requests
* Complaint count
* Transaction growth
* Blacklist match score
* Website verification
* KYC verification level

## Modeling Approach

### 1. Random Forest Baseline

A Random Forest classifier was first developed as a baseline.

The baseline achieved approximately:

**Accuracy: 87.5%**

However, its High-risk recall was only approximately:

**28.6%**

This indicated that the baseline model was missing too many High-risk merchants.

### 2. XGBoost

XGBoost was selected as the final model because it provided stronger performance for the minority High-risk class.

The model uses:

* 400 estimators
* Maximum tree depth of 5
* Learning rate of 0.05
* 85% row subsampling
* 85% feature subsampling
* Multiclass classification

Categorical features are processed using One-Hot Encoding, while numerical features are passed directly to the model.

## Class Imbalance Handling

Only **2.78%** of the dataset belongs to the High-risk class.

To address this, class-based sample weights were calculated from the training data.

Approximate training weights were:

| Class  | Weight |
| ------ | -----: |
| Low    |  0.725 |
| Medium |  0.650 |
| High   | 12.012 |

This gives substantially more importance to High-risk training examples.

## Validation-Based Threshold Tuning

Instead of relying only on the default highest-probability class prediction, a separate validation set was used to tune the High-risk decision threshold.

The selected threshold was:

**0.30**

The rule is:

> If the predicted High-risk probability is at least 30%, classify the merchant as High-risk.

The threshold was selected using the validation set while requiring High-risk recall to remain at least approximately 85%.

The final test set was kept untouched during threshold selection.

## Final Model Performance

On the untouched test set, the XGBoost model achieved:

**Accuracy: 89.6%**

Default multiclass High-risk performance:

* Precision: **67%**
* Recall: **57%**
* F1-score: **62%**

After applying the tuned 30% High-risk threshold:

* High-risk precision: **64.7%**
* High-risk recall: **78.6%**
* True positives: **11**
* False positives: **6**
* False negatives: **3**

The threshold improves the model's ability to identify High-risk merchants, which is particularly important for a risk-screening use case.

## Important Features

The final XGBoost model identified several influential features.

Top features included:

1. Previous suspension flag
2. Basic KYC verification
3. Chargeback ratio
4. Multiple bank accounts flag
5. Refund ratio
6. Full KYC verification
7. Settlement delay requests
8. Website verification
9. Registered state
10. Enhanced KYC verification
11. Crypto/Forex-adjacent business category
12. Complaint count
13. Transaction growth rate
14. Business age
15. Registered state — Maharashtra

These results suggest that historical compliance behavior, transaction disputes, KYC status, banking patterns, and merchant behavior are important signals for risk classification.

## Prediction System

The project includes an interactive command-line prediction system:

```text
src/predict.py
```

It collects merchant information from the user, sends the information through the trained preprocessing and XGBoost pipeline, and returns:

* Predicted risk level
* Low probability
* Medium probability
* High probability
* High-risk threshold
* Recommended action

Example recommendations:

| Risk   | Recommendation                  |
| ------ | ------------------------------- |
| Low    | Standard monitoring             |
| Medium | Enhanced monitoring recommended |
| High   | Manual review required          |

## Project Structure

```text
merchant-risk-scoring/
│
├── data/
│   └── merchant_risk_dataset.csv
│
├── notebook/
│
├── outputs/
│   ├── 01_risk_label_distribution.png
│   ├── 02_risk_by_category.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_risk_score_correlations.png
│   ├── 05_chargeback_by_risk.png
│   ├── 06_refund_by_risk.png
│   ├── 07_risk_score_distribution.png
│   └── feature_importance.csv
│
├── src/
│   ├── eda.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── merchant_risk_xgboost.pkl
│
├── venv/
│
├── README.md
└── requirements.txt
```

## How to Run

### 1. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Run EDA

```powershell
python src/eda.py
```

### 4. Train the model

```powershell
python src/train_model.py
```

The trained model will be saved as:

```text
models/merchant_risk_xgboost.pkl
```

### 5. Run merchant prediction

```powershell
python src/predict.py
```

The program will interactively ask for merchant information and provide a risk assessment.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib

## Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Exploratory Data Analysis
* Data preprocessing
* Categorical feature encoding
* Train/validation/test splitting
* Stratified sampling
* Class imbalance
* Sample weighting
* Random Forest
* XGBoost
* Multiclass classification
* Probability-based classification
* Decision-threshold tuning
* Target leakage prevention
* Confusion matrices
* Precision, recall, and F1-score
* Feature importance
* Model serialization
* Building an inference/prediction pipeline

## Business Impact

A merchant risk scoring system can help payment and financial platforms prioritize operational attention.

For example:

* **Low-risk merchants** can continue with standard monitoring.
* **Medium-risk merchants** can receive enhanced monitoring.
* **High-risk merchants** can be routed for manual review.

The threshold-based approach allows the business to adjust the model toward higher High-risk detection when missing a risky merchant is more costly than conducting additional reviews.

## Future Improvements

Potential next steps include:

* SHAP-based model explainability
* Probability calibration
* Hyperparameter optimization
* Cross-validation
* Model monitoring and drift detection
* REST API deployment
* Web dashboard for risk analysts
* Automated review queues
* Cost-sensitive threshold optimization
* Real-time transaction-level risk scoring

## Conclusion

This project demonstrates an end-to-end merchant risk classification workflow, from exploratory analysis and leakage prevention to model training, threshold optimization, evaluation, and interactive prediction.

The final XGBoost model achieved **89.6% test accuracy**, while threshold tuning improved the detection of High-risk merchants to **78.6% recall** on the final test set.

The project is designed as a practical demonstration of how machine learning can support merchant onboarding, monitoring, and risk-review workflows.
