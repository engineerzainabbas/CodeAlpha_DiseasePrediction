# 🏥 Disease Prediction from Medical Data
### CodeAlpha Machine Learning Internship — Task 4

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-red?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📌 Overview

A machine learning project that predicts the likelihood of **3 critical diseases** from patient medical data using multiple classification algorithms. Built as part of the **CodeAlpha ML Internship Program**.

| Disease | Dataset | Samples | Features |
|---|---|---|---|
| ❤️ Heart Disease | Cleveland UCI (Kaggle) | 1,025 | 13 |
| 🩸 Diabetes | Pima Indians (Kaggle) | 768 | 8 |
| 🔬 Breast Cancer | Wisconsin UCI (Kaggle) | 569 | 30 |

---

## 🤖 ML Algorithms Used

| Algorithm | Type |
|---|---|
| Logistic Regression | Linear |
| Random Forest | Ensemble |
| Support Vector Machine (SVM) | Kernel-based |
| Gradient Boosting | Ensemble |
| XGBoost | Boosting |

---

## 📊 Model Performance (ROC-AUC)

| Model | Heart Disease | Diabetes | Breast Cancer |
|---|---|---|---|
| Logistic Regression | 0.91 | 0.83 | 0.98 |
| Random Forest | 0.96 | 0.87 | 0.99 |
| SVM | 0.93 | 0.84 | 0.98 |
| Gradient Boosting | 0.95 | 0.86 | 0.99 |
| **XGBoost** | **0.97** | **0.88** | **0.99** |

---

## 🗂️ Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── 📁 data/
│   ├── heart.csv               # Heart Disease dataset
│   ├── diabetes.csv            # Pima Indians Diabetes dataset
│   └── breast_cancer.csv       # Wisconsin Breast Cancer dataset
│
├── 📁 models/                  # Saved trained models (.pkl)
│   ├── heart_model.pkl
│   ├── diabetes_model.pkl
│   ├── breast_cancer_model.pkl
│   └── feature_info.json
│
├── 📓 CodeAlpha_DiseasePrediction.ipynb   # Training notebook
├── 🖥️  app_disease.py                     # Streamlit dashboard
├── 📄 requirements_disease.txt            # Dependencies
└── 📄 README.md
```
