"""Local and global SHAP explanations in fraud probability space."""
from pathlib import Path
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]


def explain_local(model, row, background):
    background = background.sample(min(20, len(background)), random_state=42)
    masker = shap.maskers.Independent(background)
    explainer = shap.PermutationExplainer(lambda values: model.predict_proba(pd.DataFrame(values, columns=background.columns))[:, 1], masker)
    result = explainer(row, max_evals=2 * len(background.columns) + 1)
    values = np.asarray(result.values).reshape(-1)
    return {"base_value": float(np.asarray(result.base_values).reshape(-1)[0]),
            "values": values.tolist(), "features": row.iloc[0].to_dict(),
            "positive": sorted([{"feature": k, "value": float(row.iloc[0][k]), "shap": float(v)} for k, v in zip(row.columns, values) if v > 0], key=lambda x: -x["shap"])[:8],
            "negative": sorted([{"feature": k, "value": float(row.iloc[0][k]), "shap": float(v)} for k, v in zip(row.columns, values) if v < 0], key=lambda x: x["shap"])[:8]}


def build_global_explanation(model, X_train, X_test):
    out = ROOT / "artifacts" / "shap"; out.mkdir(parents=True, exist_ok=True)
    background = X_train.sample(min(15, len(X_train)), random_state=42)
    sample = X_test.sample(min(40, len(X_test)), random_state=43)
    explainer = shap.PermutationExplainer(lambda values: model.predict_proba(pd.DataFrame(values, columns=X_train.columns))[:, 1], shap.maskers.Independent(background))
    values = explainer(sample, max_evals=2 * len(X_train.columns) + 1)
    plt.figure(); shap.summary_plot(values.values, sample, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(out / "summary.png", dpi=150, bbox_inches="tight"); plt.close()
    mean_abs = np.abs(values.values).mean(axis=0)
    pd.DataFrame({"feature": X_train.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False).to_csv(out / "global_importance.csv", index=False)
