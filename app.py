from pathlib import Path
import io, json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, send_from_directory, url_for
from src.data_preprocessing import FEATURES, load_dataset
from src.shap_explainer import explain_local

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset" / "creditcard.csv"
RESULTS = ROOT / "models" / "model_results.json"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
app = Flask(__name__)
app.secret_key = "fraud-detection-local-app"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

def load_results():
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else None

def load_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

def stats():
    if DATA.exists():
        frame = load_dataset(DATA)
        return {"total": len(frame), "genuine": int((frame.Class == 0).sum()), "fraud": int((frame.Class == 1).sum()), "fraud_percent": float(frame.Class.mean()*100), "genuine_percent": float((1-frame.Class.mean())*100)}
    return None

@app.context_processor
def inject_status():
    return {"trained": MODEL_PATH.exists() and RESULTS.exists()}

@app.route("/")
def index(): return render_template("index.html", stats=stats())

@app.route("/about")
def about(): return render_template("about.html")

@app.route("/artifacts/<path:filename>")
def artifact(filename): return send_from_directory(ROOT / "artifacts", filename)

def predict_one(row):
    model = load_model()
    if model is None: raise RuntimeError("Models are not trained yet. Add the dataset and run python src/train_models.py.")
    probability = float(model.predict_proba(row[FEATURES])[:, 1][0])
    prediction = int(probability >= 0.5)
    explanation = None
    if DATA.exists():
        data = load_dataset(DATA)[FEATURES]
        explanation = explain_local(model, row[FEATURES], data.sample(n=min(20, len(data)), random_state=42))
    return {"prediction": prediction, "label": "Fraudulent" if prediction else "Genuine", "probability": probability, "explanation": explanation}

@app.route("/detect", methods=["GET", "POST"])
def detect():
    result = None
    if request.method == "POST":
        try:
            row = pd.DataFrame([{name: float(request.form[name]) for name in FEATURES}], columns=FEATURES)
            if not np.isfinite(row.to_numpy()).all(): raise ValueError("Values must be finite numbers.")
            result = predict_one(row)
        except (KeyError, ValueError): flash("Enter a valid numeric value for every feature.", "error")
        except RuntimeError as exc: flash(str(exc), "error")
    return render_template("detect.html", features=FEATURES, result=result)

@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True) or request.form
    try:
        row = pd.DataFrame([{name: float(payload[name]) for name in FEATURES}], columns=FEATURES)
        if not np.isfinite(row.to_numpy()).all(): raise ValueError("Values must be finite numbers.")
        return jsonify(predict_one(row))
    except (KeyError, TypeError, ValueError) as exc: return jsonify(error="Provide numeric values for all 30 required features.", detail=str(exc)), 400
    except RuntimeError as exc: return jsonify(error=str(exc)), 503

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    upload = request.files.get("file")
    if not upload or not upload.filename.lower().endswith(".csv"):
        flash("Choose a CSV file containing the required feature columns.", "error"); return redirect(url_for("detect"))
    try:
        frame = pd.read_csv(upload)
        missing = set(FEATURES) - set(frame.columns)
        if missing: raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
        if len(frame) > 10000: raise ValueError("Upload limit is 10,000 transactions.")
        X = frame[FEATURES].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(X.to_numpy()).all(): raise ValueError("Input contains missing or non-finite values.")
        model = load_model()
        if model is None: raise RuntimeError("Models are not trained. Run the training command first.")
        probs = model.predict_proba(X)[:, 1]
        output = pd.DataFrame({"transaction_number": range(1, len(X)+1), "prediction": ["Fraudulent" if p >= .5 else "Genuine" for p in probs], "fraud_probability": probs})
        return render_template("detect.html", features=FEATURES, result=None, batch=output.to_dict("records"), download=output.to_csv(index=False))
    except (ValueError, RuntimeError, pd.errors.ParserError) as exc:
        flash(str(exc), "error"); return redirect(url_for("detect"))

@app.route("/batch-download", methods=["POST"])
def batch_download():
    return send_file(io.BytesIO(request.form.get("csv", "").encode()), mimetype="text/csv", as_attachment=True, download_name="fraud_predictions.csv")

@app.route("/dashboard")
def dashboard():
    results = load_results()
    path = ROOT / "artifacts/shap/global_importance.csv"
    importance = pd.read_csv(path).head(10).to_dict("records") if path.exists() else []
    return render_template("dashboard.html", stats=stats(), results=results, importance=importance)

@app.route("/comparison")
def comparison(): return render_template("comparison.html", results=load_results())

@app.errorhandler(413)
def too_large(_):
    flash("Upload exceeds the 8 MB limit.", "error"); return redirect(url_for("detect"))

@app.errorhandler(500)
def server_error(_): return render_template("error.html", message="Something went wrong while processing the request."), 500

if __name__ == "__main__": app.run(debug=False)