# 🔧 Predictive Maintenance using Sensor Data

## 📌 Project Overview

This project focuses on building a **machine learning model to predict
equipment failure** using sensor data.\
The objective is to enable **predictive maintenance**, reducing costly
equipment replacement and minimizing operational downtime.

The dataset consists of transformed (ciphered) sensor features collected
from industrial equipment.\
The target variable indicates whether a machine failure occurred (1) or
not (0).

------------------------------------------------------------------------

## 🎯 Business Problem

Equipment replacement is significantly more expensive than preventive
repairs, and unexpected downtime leads to operational losses.

The goal of this project is to:

-   Predict potential equipment failures in advance\
-   Minimize **false negatives** (missed failures)\
-   Reduce maintenance and replacement costs\
-   Enable data-driven maintenance scheduling

Because failure events are rare (\~5--6%), this is a **highly imbalanced
classification problem**.

------------------------------------------------------------------------

## 📊 Dataset Description

-   **Training Data:** 20,000 rows\
-   **Test Data:** 5,000 rows\
-   **Features:** 40 numerical sensor variables (V1 -- V40)\
-   **Target:** Binary (0 = No Failure, 1 = Failure)

### Challenges:

-   Imbalanced dataset (\~5.6% failures)
-   Missing values in V1 and V2
-   Ciphered feature names (limited interpretability)

------------------------------------------------------------------------

## 🛠 Data Preprocessing

-   Missing values handled using **KNN Imputation (k=5)**
-   Stratified train-validation-test split
-   Class imbalance handled using:
    -   SMOTE (oversampling)
    -   Random undersampling
-   Final pipeline includes preprocessing + sampling + model

------------------------------------------------------------------------

## 🤖 Models Implemented

Multiple supervised learning models were evaluated:

-   Decision Tree
-   Random Forest
-   Gradient Boosting Machine (GBM)
-   AdaBoost
-   XGBoost
-   Logistic Regression
-   Ridge Classifier

### ✅ Final Selected Model

**Gradient Boosting Classifier (GBM)** with undersampling.

Chosen because it provided the best balance between: - High Recall -
Reduced overfitting - Strong generalization performance

------------------------------------------------------------------------

## 📈 Training Strategy

-   Stratified K-Fold Cross Validation (k=5)
-   Hyperparameter tuning using:
    -   RandomizedSearchCV
    -   GridSearchCV
-   Primary metric optimized: **Recall**
    -   Business priority: avoid missing failures

------------------------------------------------------------------------

## 📊 Model Performance

Final pipeline evaluation (held-out test set):

-   Accuracy: 0.964\
-   Recall: 0.625\
-   Precision: 0.700\
-   F1 Score: 0.660

Recall was prioritized due to the high cost of undetected failures.

------------------------------------------------------------------------

## 🔄 Model Pipeline

Conceptual workflow:

Raw Data → Missing Value Imputation (KNN) → Class Rebalancing
(Undersampling) → Gradient Boosting Model → Evaluation

Pipeline built using `sklearn.pipeline.Pipeline` for reproducibility and
deployment readiness.

------------------------------------------------------------------------

## 🚀 Deployment Considerations

-   Encapsulated preprocessing + model in a single pipeline
-   Ready for serialization (e.g., joblib)
-   Suitable for batch or API-based deployment
-   Future improvements:
    -   Model monitoring
    -   Drift detection
    -   CI/CD automation

------------------------------------------------------------------------

## 🔎 Key Insights

Top predictive features: - V18 - V36 - V15 - V3 - V26

Certain threshold combinations significantly increase failure
probability, enabling actionable maintenance triggers.

------------------------------------------------------------------------

## 📦 Tech Stack

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   XGBoost
-   Imbalanced-learn
-   Matplotlib

------------------------------------------------------------------------

## 📚 Future Improvements

-   Cost-sensitive learning optimization
-   More labeled failure data
-   Time-series feature engineering
-   Production-grade monitoring pipeline

------------------------------------------------------------------------

## 👤 Author

Promise Ekpo\
Data Scientist \| Automation Analyst \| Public Health & AI Enthusiast

------------------------------------------------------------------------

## ⭐ If you find this useful

Give this repository a star ⭐ and feel free to fork or contribute!
