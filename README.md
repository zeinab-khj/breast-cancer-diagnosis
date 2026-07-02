# 🩺 Breast Cancer Diagnosis using Machine Learning

![Python](https://img.shields.io/badge/python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/status-completed-green)

---

## 📌 Project Overview

This project is an **end-to-end Machine Learning pipeline** for breast cancer diagnosis.
---
## 📌 Key Takeaways

- End-to-end ML pipeline implemented
- Strong model performance achieved using SVM + Optuna
- Full experiment tracking with MLflow
- Model interpretability using SHAP


---

## 🎯 Problem Statement

This project aims to:
- Build a robust classification model
- Compare multiple ML algorithms
- Optimize the best-performing model
- Provide model interpretability using SHAP

---

## 🔄 Workflow

```text
Data Loading
      ↓
Train / Test Split
      ↓
EDA (Exploration & Insights)
      ↓
Baseline Models Training
(Logistic Regression, SVM, Random Forest, XGBoost)
      ↓
Best Model Selection (SVM)
      ↓
Hyperparameter Tuning (Optuna)  + MLflow tracking
      ↓
Final Model Training (SVM + best params) + MLflow log model
      ↓
Evaluation (Accuracy, Precision, Recall, F1)
      ↓
Model Explainability (SHAP)
      ↓
Model Saving (joblib) + MLflow artifact
```
## 📂 Project Structure

```text
breast-cancer-project/
│
├── notebooks/              # EDA + Experiments
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
│
├── src/                    # Core ML pipeline
│   ├── preprocess.py
│   ├── train.py
│   ├── tune.py
│   └── evaluate.py
│
├── models/                 # Trained models
│   └── save_model.py
│
├── scripts/               # Entry point
│   └── run_training.py
│
├── requirements.txt
└── README.md
```


---

## 📊 Exploratory Data Analysis (EDA)

Key insights from EDA:

- No missing values in dataset
- Clear separation between classes
- Strong correlation among certain features
- Some redundant features identified

---

## 🤖 Model Benchmarking

Several machine learning models were evaluated:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score

✔ **Best Model: SVM**
**why SVM?**
SVM outperformed Logistic Regression,
Random Forest and XGBoost on this dataset,
achieving the highest cross-validation score while maintaining strong generalization.

---

## ⚙️ Hyperparameter Optimization

The SVM model was optimized using **Optuna** with 5-fold cross-validation.

### Tuned parameters:
- C
- Kernel
- Gamma

✔ Optimization improved model stability and generalization.

---

## 📈 Model Evaluation

Final model performance:

| Metric     | Score |
|------------|------:|
| Accuracy   | 0.98  |
| Precision  | 0.97  |
| Recall     | 0.98  |
| F1-score   | 0.98  |

The optimized SVM achieved the best balance between precision and recall compared with all baseline models.

### Confusion Matrix
<img width="498" height="433" alt="Confuion_Matrix" src="https://github.com/user-attachments/assets/ed71f219-50f1-479a-873a-a6365f2346cb" />


### ROC Curve
<img width="1001" height="701" alt="ROC_Curve" src="https://github.com/user-attachments/assets/e793f47c-5571-4081-8bf8-28186b1b3258" />


---

## 🧠 Model Explainability

SHAP was used to interpret model predictions.

Key insights:
- Most important features: tumor radius, texture, and perimeter
- Model decisions are clinically interpretable

### SHAP Summary Plot
<img width="762" height="940" alt="SHAP_Summary" src="https://github.com/user-attachments/assets/0e1ac821-4510-403d-877f-810e2830bcf2" />


---

## 📈 MLflow Tracking

All experiments were tracked using MLflow:
- Hyperparameters
- Metrics
- Model artifacts

To launch UI:

```bash
mlflow ui
```
---

## 💾 Model Saving

The final trained model is saved using `joblib`: models/svm_optuna.pkl


This allows direct inference without retraining.

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Optuna
- SHAP
- MLflow
- Matplotlib
- Seaborn




---
## 🚀 How to Run

```bash
pip install -r requirements.txt

python scripts/run_training.py
```
---


## 🎯 Future Work


- Deploy using FastAPI
- Dockerize the project
- Compare with LightGBM
- Perform feature selection

---


