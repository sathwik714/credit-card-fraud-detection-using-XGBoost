# 💳 Credit Card Fraud Detection using XGBoost

This project uses the **XGBoost** machine learning algorithm to detect fraudulent credit card transactions on a real-world, anonymized dataset. It includes data preprocessing, model training, evaluation, and performance comparison against other classifiers such as Logistic Regression and Random Forests.

---

## 📊 Project Overview

Credit card fraud detection is a critical application of machine learning in the financial industry. This project demonstrates how **Gradient Boosting Machines (GBMs)**, particularly **XGBoost**, can be leveraged to build a powerful classification model capable of identifying fraudulent transactions with high accuracy.

---

## 🔍 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Description**:
  - 284,807 transactions
  - 492 frauds (highly imbalanced)
  - Features are anonymized (PCA-transformed), with `Time`, `Amount`, and `Class` (target variable)

---

## 🛠 Features

- Data preprocessing and scaling
- Handling class imbalance using undersampling/SMOTE
- XGBoost model training and tuning
- Performance comparison with:
  - Logistic Regression
  - Random Forest
- Evaluation metrics: Accuracy, Precision, Recall, F1-score, AUC
- Confusion matrix and ROC curve visualization

---

## 🧪 Model Performance

| Model               | Accuracy | Precision | Recall | F1-score | AUC     |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression| 0.96     | 0.81      | 0.66   | 0.73     | 0.90    |
| Random Forest       | 0.97     | 0.88      | 0.76   | 0.82     | 0.94    |
| **XGBoost**         | **0.98** | **0.91**  | **0.81** | **0.86** | **0.97** |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
