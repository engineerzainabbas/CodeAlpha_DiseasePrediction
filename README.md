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

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction
```

### 2. Install dependencies
```bash
pip install -r requirements_disease.txt
```

### 3. Download datasets
Download from Kaggle and place in `data/` folder:
- [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) → `heart.csv`
- [Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) → `diabetes.csv`
- [Breast Cancer](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) → `breast_cancer.csv`

### 4. Train models
```bash
jupyter notebook CodeAlpha_DiseasePrediction.ipynb
```
Run all cells — models will be saved to `models/`

### 5. Launch dashboard
```bash
streamlit run app_disease.py
```

---

## 🖥️ Dashboard Features

- **🔍 Real-time Prediction** — Enter patient data via sliders → instant disease prediction
- **📊 Confidence Score** — Shows probability for each class
- **📈 Data Explorer** — Visualize dataset distributions and statistics
- **🏗️ Model Performance** — Compare all 5 algorithms side by side
- **3 Disease Tabs** — Switch between Heart, Diabetes, Breast Cancer

---

## 📓 Notebook Contents

1. **Data Loading** — Load all 3 datasets from local CSV files
2. **EDA** — Class distribution, missing values, correlation heatmaps
3. **Preprocessing** — Train/test split, StandardScaler pipeline
4. **Model Training** — 5 algorithms × 3 diseases = 15 models with 5-fold CV
5. **Evaluation** — Accuracy, F1, ROC-AUC, Confusion Matrix, ROC Curves
6. **Feature Importance** — Random Forest feature rankings
7. **Save Models** — Export best model per disease as `.pkl`

---

## 📦 Dependencies

```
streamlit>=1.32.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
joblib>=1.3.0
```

---

## 🔍 Key Features of the ML Pipeline

- ✅ **Stratified K-Fold Cross Validation** (5 folds)
- ✅ **StandardScaler** inside Pipeline (no data leakage)
- ✅ **ROC-AUC** as primary evaluation metric
- ✅ **Best model** auto-selected per disease
- ✅ **Feature Importance** visualization (Random Forest)
- ✅ **Confusion Matrix** for all best models

---

## 📸 Screenshots

### Dashboard — Prediction
> Enter patient data → get real-time disease prediction with confidence score

### Dashboard — Data Explorer
> Visualize dataset distributions, class balance, and feature statistics

### Notebook — Model Comparison
> Side-by-side comparison of all 5 algorithms across 3 diseases

---

## 👨‍💻 Author

**Zain Abbas**
- 🏢 CodeAlpha ML Internship 2024
- 🔗 [LinkedIn](https://linkedin.com/in/engineerzainabbas
- 🐙 [GitHub](https://github.com/engineerzainabbas

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- [CodeAlpha](https://www.codealpha.tech) — Internship Program
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/) — Datasets
- [Kaggle](https://www.kaggle.com) — Dataset hosting
- [Scikit-learn](https://scikit-learn.org) — ML algorithms
- [Streamlit](https://streamlit.io) — Dashboard framework
