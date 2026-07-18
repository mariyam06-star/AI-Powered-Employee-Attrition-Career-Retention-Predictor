# AI-Powered Employee Attrition & Career Retention Predictor
### Women in the Workforce

An end-to-end Machine Learning project that predicts employee attrition risk, explains *why* an employee is at risk, and turns that risk into concrete career-support recommendations — with a specific lens on the structural barriers women face at work (overtime culture, slower promotions, limited flexibility, career breaks).

Built as part of the **AI Career for Women (AICW)** program, Engg-HUB track.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Dashboard](#dashboard)
- [Team](#team)
- [References](#references)
- [License](#license)

---

## Problem Statement

Most HR systems only learn that an employee was a flight risk **after** they've already resigned — by which point it's too late to help. This challenge is particularly significant for women in the workforce, who often face compounding structural pressures that raise their attrition risk.

This project closes that gap with a proactive, explainable, data-driven system that flags at-risk employees early and recommends specific retention actions instead of handing back a black-box score.

## Features

- 🔍 **Exploratory Data Analysis** — attrition patterns by gender, overtime, income, tenure, and promotion history
- ⚖️ **Class imbalance handling** with SMOTE (attrition is a ~16% minority class)
- 🤖 **Two ML models compared** — Random Forest and XGBoost
- 📊 **Full evaluation suite** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix, ROC curve
- ⭐ **Explainability** — feature importance ranking to surface top attrition drivers
- 🖥️ **Interactive Streamlit dashboard** — live risk prediction from an employee profile, with career-support recommendations
- 👩 **Gender-segmented analysis** highlighting where targeted retention support has the most impact

## Dataset

**IBM HR Analytics Employee Attrition & Performance Dataset**
1,470 employee records · 35 features · publicly available, no additional data collection required.

Source: [Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (originally published by IBM)

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Imbalance handling | imbalanced-learn (SMOTE) |
| Model persistence | Joblib |
| Dashboard | Streamlit |

## Project Structure

```
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── notebooks/
│   └── Employee_Attrition_Prediction.ipynb
├── models/
│   ├── employee_attrition_model.pkl
│   ├── feature_names.pkl
│   └── label_encoder.pkl
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── docs/
│   └── Notebook_Documentation.docx
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.x
- 8 GB RAM minimum

### Installation

```bash
git clone https://github.com/<your-username>/employee-attrition-predictor.git
cd employee-attrition-predictor
pip install -r requirements.txt
```

`requirements.txt` should include:
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
streamlit
joblib
```

### Run the notebook
```bash
jupyter notebook notebooks/Employee_Attrition_Prediction.ipynb
```

### Run the dashboard
```bash
streamlit run dashboard/app.py
```

## Model Pipeline

1. **Data Collection** — load the IBM HR Analytics dataset
2. **EDA** — attrition patterns overall and by gender vs. OverTime, Work-Life Balance, Income, Promotions
3. **Preprocessing** — one-hot encoding, `StandardScaler`, 80/20 stratified train-test split
4. **Class Balancing** — SMOTE applied to training data only (no test-set leakage)
5. **Model Training** — Random Forest and XGBoost classifiers
6. **Evaluation** — Accuracy, Precision, Recall, F1-score, ROC-AUC, confusion matrix
7. **Explainability** — feature importance for top attrition drivers
8. **Export** — best model, feature list, and label encoder saved for the dashboard

## Results

| Metric | Random Forest | XGBoost |
|---|---|---|
| Accuracy | see `notebooks/` output | see `notebooks/` output |
| Precision | see `notebooks/` output | see `notebooks/` output |
| Recall | see `notebooks/` output | see `notebooks/` output |
| F1 Score | see `notebooks/` output | see `notebooks/` output |
| ROC-AUC | see `notebooks/` output | see `notebooks/` output |

> Run the notebook to reproduce these numbers on your machine — exact values depend on the random seed and environment.

## Dashboard

The Streamlit dashboard includes:
- **Home** — project overview and key stats
- **Dataset Overview** — EDA visualizations
- **Employee Attrition Prediction** — enter an employee profile, get a live risk score
- **Model Performance** — metrics and comparison charts
- **About Project** — team and methodology

## Team

| Name | Role |
|---|---|
| Mariyam Yusuf Sidat | Team Lead |
| Patil Trupti Rajubhai | Team Member |

**Guide:** Jyoti Prajapti, Internship Mentor
**Institution:** R.N.G. Patel Institute of Technology — Information Technology

## References

- [IBM HR Analytics Employee Attrition Dataset — Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Imbalanced-learn (SMOTE) Documentation](https://imbalanced-learn.org/stable/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is licensed under the [MIT License](LICENSE) — free to use, modify, and distribute with attribution.
