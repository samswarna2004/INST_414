# sprint3_models.py
"""
Sprint 3: Classification models to predict whether a player makes the cut.

Models:
- Logistic Regression
- Random Forest
- Gradient Boosting

Outputs:
- Saved models in models/
- Metrics CSV in results/metrics_sprint3.csv
- Classification reports in results/
- Figures in reports/figures/
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay,
    classification_report
)

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"

MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(exist_ok=True)

csv_path = DATA / "STAGE_3_SPRINT_2.csv"
print(">>> Using CSV:", csv_path)

# ---------- load data ----------
df = pd.read_csv(csv_path)
if df.shape[1] == 1:
    df = pd.read_csv(csv_path, sep=";")

# clean column names
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.lower()
      .str.replace(r"[^a-z0-9]+", "_", regex=True)
      .str.strip("_")
)

print(">>> Columns:", list(df.columns))

# target + features
target = "made_cut"   # must be 0/1
feature_cols = ["sg_ott", "sg_app", "sg_arg", "sg_putt", "sg_total"]

df = df.dropna(subset=[target] + feature_cols).copy()
X = df[feature_cols]
y = df[target].astype(int)

print(">>> Final modeling shape:", X.shape)

# ---------- train/val/test split ----------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(">>> Train size:", X_train.shape[0],
      "Val size:", X_val.shape[0],
      "Test size:", X_test.shape[0])

# ---------- helper: train + eval ----------
def train_and_eval(model, name):
    """Fit model, evaluate on val/test, save artifacts."""
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)

    # save model
    model_path = MODELS / f"{name}.joblib"
    joblib.dump(model, model_path)
    print(f"Saved model -> {model_path}")

    # val metrics
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = (val_proba >= 0.5).astype(int)

    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    def metrics(y_true, y_hat, y_score):
        return {
            "accuracy": accuracy_score(y_true, y_hat),
            "precision": precision_score(y_true, y_hat),
            "recall": recall_score(y_true, y_hat),
            "f1": f1_score(y_true, y_hat),
            "roc_auc": roc_auc_score(y_true, y_score),
        }

    val_metrics = metrics(y_val, val_pred, val_proba)
    test_metrics = metrics(y_test, test_pred, test_proba)

    # classification report (test)
    report = classification_report(y_test, test_pred)
    report_path = RESULTS / f"classification_report_{name}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved classification report -> {report_path}")

    # confusion matrix figure (test)
    disp = ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test
    )
    disp.ax_.set_title(f"Confusion Matrix – {name}")
    fig_cm_path = FIGS / f"confmat_{name}.png"
    plt.tight_layout()
    plt.savefig(fig_cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix -> {fig_cm_path}")

    # ROC curve (test)
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"{name} (AUC = {test_metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {name}")
    plt.legend()
    fig_roc_path = FIGS / f"roc_{name}.png"
    plt.tight_layout()
    plt.savefig(fig_roc_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve -> {fig_roc_path}")

    return name, val_metrics, test_metrics


# ---------- define models ----------
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

gb = GradientBoostingClassifier(
    random_state=42
)

# ---------- train + eval all ----------
results = []

for model, name in [
    (log_reg, "logistic_regression"),
    (rf, "random_forest"),
    (gb, "gradient_boosting"),
]:
    name, val_m, test_m = train_and_eval(model, name)
    row_val = {"model": name, "split": "val", **val_m}
    row_test = {"model": name, "split": "test", **test_m}
    results.extend([row_val, row_test])

metrics_df = pd.DataFrame(results)
metrics_path = RESULTS / "metrics_sprint3.csv"
metrics_df.to_csv(metrics_path, index=False)
print("Saved metrics ->", metrics_path)

print("\nDone.")


print("Saved metrics ->", metrics_path)
print("\nDone.")


# ------------------------------------
# Save model hyperparameters (JSON)
# ------------------------------------
import json

params_path = RESULTS / "model_params_sprint3.json"
params = {
    "logistic_regression": log_reg.get_params(),
    "random_forest": rf.get_params(),
    "gradient_boosting": gb.get_params()
}

with open(params_path, "w") as f:
    json.dump(params, f, indent=2, default=str)

print("Saved params ->", params_path)


# ------------------------------------
# Feature Importance – Random Forest
# ------------------------------------
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_names = X.columns

plt.figure()
plt.title("Feature Importance – Random Forest")
plt.bar(range(len(feat_names)), importances[indices])
plt.xticks(range(len(feat_names)), feat_names[indices], rotation=45)
plt.tight_layout()

feat_fig_path = FIGS / "feature_importance_rf.png"
plt.savefig(feat_fig_path, dpi=300)
plt.close()

print("Saved feature importance ->", feat_fig_path)
