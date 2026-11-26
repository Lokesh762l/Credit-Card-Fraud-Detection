"""Train the five required classifiers on original and SMOTE training data."""
import json
import os
from pathlib import Path
import joblib
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from data_preprocessing import FEATURES, RANDOM_STATE, load_dataset, split_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "creditcard.csv"
MODELS = ROOT / "models"
PLOTS = ROOT / "artifacts" / "plots"


def model_definitions():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=RANDOM_STATE),
        "Bagging Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=30,
            n_jobs=-1, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=70, random_state=RANDOM_STATE),
    }


def build_pipeline(estimator, use_smote):
    steps = []
    if estimator.__class__.__name__ == "LogisticRegression":
        steps.append(("scaler", StandardScaler()))
    if use_smote:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def evaluate(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main():
    if not DATA.exists():
        raise SystemExit(f"Dataset not found: {DATA}. Place Kaggle creditcard.csv there first.")
    MODELS.mkdir(exist_ok=True); PLOTS.mkdir(parents=True, exist_ok=True)
    (ROOT / "artifacts" / "metrics").mkdir(parents=True, exist_ok=True)
    frame = load_dataset(DATA)
    X_train, X_test, y_train, y_test = split_dataset(frame)
    summary = {
        "dataset": {"total": len(frame), "genuine": int((frame.Class == 0).sum()),
            "fraud": int((frame.Class == 1).sum()), "fraud_percent": float(frame.Class.mean() * 100),
            "genuine_percent": float((1-frame.Class.mean()) * 100), "missing_values": int(frame.isna().sum().sum()),
            "duplicate_rows": int(frame.duplicated().sum())},
        "test_size": len(X_test), "features": FEATURES, "models": {}, "best": None,
    }
    best = None
    curve_figures = {condition: (plt.subplots(figsize=(8, 6)), plt.subplots(figsize=(8, 6)))
                     for condition in ("original", "smote")}
    for condition, use_smote in (("original", False), ("smote", True)):
        summary["models"][condition] = {}
        (MODELS / condition).mkdir(exist_ok=True)
        for name, estimator in model_definitions().items():
            print(f"Training {name} ({condition})...", flush=True)
            model = build_pipeline(estimator, use_smote)
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            summary["models"][condition][name] = metrics
            joblib.dump(model, MODELS / condition / f"{name.lower().replace(' ', '_')}.joblib")
            (roc_figure, roc_axis), (pr_figure, pr_axis) = curve_figures[condition]
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=roc_axis, name=name)
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=pr_axis, name=name)
            candidate = (metrics["pr_auc"], metrics["recall"], metrics["precision"])
            if best is None or candidate > best[0]:
                best = (candidate, condition, name, model)
    summary["best"] = {"condition": best[1], "model": best[2], "metrics": summary["models"][best[1]][best[2]]}
    joblib.dump(best[3], MODELS / "best_model.joblib")
    pd.DataFrame([{"condition": c, "model": n, **v} for c, group in summary["models"].items()
                  for n, v in group.items()]).drop(columns="confusion_matrix").to_csv(ROOT / "artifacts" / "metrics" / "model_comparison.csv", index=False)
    for condition, ((roc_figure, roc_axis), (pr_figure, pr_axis)) in curve_figures.items():
        roc_axis.set_title(f"ROC curves — {condition} training")
        roc_figure.tight_layout(); roc_figure.savefig(PLOTS / f"roc_curves_{condition}.png", dpi=150); plt.close(roc_figure)
        pr_axis.set_title(f"Precision–Recall curves — {condition} training")
        pr_figure.tight_layout(); pr_figure.savefig(PLOTS / f"pr_curves_{condition}.png", dpi=150); plt.close(pr_figure)
    for condition, group in summary["models"].items():
        for name, metrics in group.items():
            fig, ax = plt.subplots(figsize=(4, 4)); cm = np.asarray(metrics["confusion_matrix"])
            ax.imshow(cm, cmap="Blues"); ax.set(title=f"{name} — {condition}", xlabel="Predicted", ylabel="Actual")
            for (i, j), val in np.ndenumerate(cm): ax.text(j, i, str(val), ha="center", va="center")
            fig.tight_layout(); fig.savefig(PLOTS / f"cm_{condition}_{name.lower().replace(' ', '_')}.png", dpi=130); plt.close(fig)
    (MODELS / "model_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best model by PR-AUC: {best[2]} ({best[1]}), PR-AUC={best[0][0]:.4f}")


if __name__ == "__main__":
    main()
