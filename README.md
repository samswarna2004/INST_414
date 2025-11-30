# 📊 INST 414 — Sprint 3: Classification Modeling (PGA Golf Analytics)

This repository contains all code, data, models, and results for **Sprint 3** of the INST 414 course.  
The goal of this sprint was to develop, train, evaluate, and compare classification models that predict whether a PGA player **makes the cut** using strokes-gained and scoring metrics.

This repo is fully reproducible, with structured folders, documented scripts, saved model artifacts, and complete evaluation outputs.

---

## 📁 Repository Structure

```
INST_414/
│
├── data/
│   └── STAGE_3_SPRINT_2.csv          # Cleaned dataset used for Sprint 3 modeling
│
├── models/
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   └── gradient_boosting.joblib
│
├── reports/
│   └── figures/
│       ├── roc_logistic_regression.png
│       ├── roc_random_forest.png
│       ├── roc_gradient_boosting.png
│       ├── confmat_logistic_regression.png
│       ├── confmat_random_forest.png
│       ├── confmat_gradient_boosting.png
│       ├── feature_importance_rf.png
│       └── hist_strokes.png
│
├── results/
│   ├── metrics_sprint3.csv
│   ├── classification_report_logistic.txt
│   ├── classification_report_random_forest.txt
│   ├── classification_report_gradient_boosting.txt
│   └── model_params_sprint3.json
│
├── sprint2.py
├── sprint3_models.py
├── requirements.txt
└── README.md
```

---

## 🎯 Project Objective (Sprint 3)

Predict whether a PGA Tour player **makes the cut** using:

- Strokes gained metrics (OTT, APP, ARG, PUTT, Total)
- Scoring/positional features
- Tournament-level information

This is a **binary classification task** (`cut_made` = 1 or 0).

Models developed:

- **Logistic Regression**
- **Random Forest Classifier**
- **Gradient Boosting Classifier**

---

## 🧪 How to Run the Sprint 3 Code

### **1️⃣ Create and activate your virtual environment**
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Sprint 3 modeling script**
Make sure you are inside the repo directory:

```bash
python sprint3_models.py
```

### **4️⃣ Outputs will be generated automatically**

After the script runs, you will see:

✔ Trained models saved in `models/`  
✔ ROC curves, confusion matrices, feature importance images in `reports/figures/`  
✔ Accuracy, precision, recall, F1, ROC-AUC in `results/metrics_sprint3.csv`  
✔ Classification reports in `results/`  
✔ Hyperparameter JSON saved in `results/model_params_sprint3.json`

---

## 📈 Modeling Summary (Sprint 3 Highlights)

- **Logistic Regression** provides interpretability and strong baseline performance.
- **Random Forest** performs well with non-linear relationships and provides feature importance.
- **Gradient Boosting** typically gives the most accurate model due to sequential error correction.

**Feature Importance (RF)**  
Top predictors of making the cut include:

1. Strokes Gained: Total  
2. Strokes Gained: Approach  
3. Strokes Gained: Off the Tee  

These align strongly with golf analytics research.

---

## 📊 Key Evaluation Metrics (from metrics_sprint3.csv)

| Model               | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------------------|----------|-----------|--------|----|---------|
| Logistic Regression | ...      | ...       | ...    | ...| ...     |
| Random Forest       | ...      | ...       | ...    | ...| ...     |
| Gradient Boosting   | ...      | ...       | ...    | ...| ...     |


---

## 🚀 Reproducibility

This entire repository is structured to allow any user (including graders) to:

- Install dependencies  
- Run the modeling script  
- Recreate all figures, reports, and models exactly  

All required files are version-controlled in GitHub.

---

## 📬 Contact

If you have questions or need clarification, feel free to reach out.
