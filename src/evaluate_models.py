"""Regenerate ROC and precision-recall figures from saved fitted pipelines."""
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from data_preprocessing import load_dataset, split_dataset
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "bagging_classifier", "adaboost"]


def main():
    frame = load_dataset(ROOT / "dataset" / "creditcard.csv")
    _, X_test, _, y_test = split_dataset(frame)
    output = ROOT / "artifacts" / "plots"
    output.mkdir(parents=True, exist_ok=True)
    for condition in ("original", "smote"):
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
        for name in MODEL_NAMES:
            model = joblib.load(ROOT / "models" / condition / f"{name}.joblib")
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=name.replace("_", " ").title())
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr, name=name.replace("_", " ").title())
        ax_roc.set_title(f"ROC curves — {condition} training")
        fig_roc.tight_layout(); fig_roc.savefig(output / f"roc_curves_{condition}.png", dpi=150); plt.close(fig_roc)
        ax_pr.set_title(f"Precision–Recall curves — {condition} training")
        fig_pr.tight_layout(); fig_pr.savefig(output / f"pr_curves_{condition}.png", dpi=150); plt.close(fig_pr)


if __name__ == "__main__":
    main()
