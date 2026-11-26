# Credit Card Fraud Detection

A Flask application and reproducible machine-learning study for the Kaggle Credit Card Fraud Detection dataset. It compares five classifiers, measures the effect of class imbalance and training-only SMOTE, and provides transaction predictions, evaluation charts, and SHAP explanations.

## What the project does

- Trains Logistic Regression, Decision Tree, Random Forest, Bagging, and AdaBoost classifiers.
- Compares training on the original imbalanced data with training where SMOTE resamples the training portion.
- Evaluates every model on the same untouched, stratified 20% test split.
- Selects a saved model by PR-AUC, then recall and precision, and predicts fraud when the probability is at least 0.50.
- Provides single-row and batch CSV predictions in the web application.

## Dataset

Download `creditcard.csv` from the Kaggle Credit Card Fraud Detection dataset and put it at `dataset/creditcard.csv`. This dataset has `Time`, `V1` through `V28`, `Amount`, and `Class`. `Class` is the target (0 = genuine, 1 = fraud). `V1` through `V28` are anonymized PCA components, not ordinary transaction details such as merchant or location. A person generally cannot derive them from a regular transaction receipt.

The dataset is excluded from version control. In a fresh checkout, create the `dataset` directory and place the downloaded file there before training.

## Setup (Windows PowerShell)

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the virtual environment's Python directly as shown below instead.

## Train the models and start the application

Training requires `dataset/creditcard.csv` and creates model and evaluation artifacts:

```powershell
.\.venv\Scripts\python.exe src\train_models.py
```

Then start the web application:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000> in a browser. Keep the terminal running while using the application. Stop it with Ctrl+C.

The split uses `test_size=0.2`, `random_state=42`, and `stratify=y` in `src/data_preprocessing.py`. Models fit on the 80% training portion. SMOTE is inside the training pipeline, so it only resamples training data. The 20% test portion is not used for fitting or resampling; it is used for metrics, ROC and precision-recall curves, and representative SHAP examples. This makes the reported test performance an estimate of performance on held-out data.

Training writes fitted models to `models/`, metrics to `artifacts/metrics/`, plots to `artifacts/plots/`, and SHAP outputs to `artifacts/shap/`. Run training again after changing the dataset or training code.

## Try the two manual test examples

The `manual_test_data/` folder contains two one-row CSV files:

- `test_transaction_1.csv`
- `test_transaction_2.csv`

Each file has the 30 required input columns (`Time`, `V1`-`V28`, `Amount`) and deliberately omits `Class`, so the label is not supplied to the predictor. The examples were selected from the held-out test split; one is genuine and one is fraudulent, but the files do not reveal which. They are useful for trying the prediction flow, not for replacing the full test-set evaluation.

To get a prediction:

1. Train the models and start the app using the commands above.
2. Open the **Detection** page.
3. In **Batch CSV**, choose one of the two files from `manual_test_data/` and click **Predict CSV**.
4. Read the predicted class and fraud probability shown in the results. The app does not reveal the example's true label.

The batch uploader accepts CSVs with the required feature columns and ignores extra columns such as `Class`. Do not use the manual form's zero defaults as a real transaction; they are placeholders. For a real transaction, this dataset alone cannot translate normal transaction details into the anonymized PCA fields.

## Application pages

- **Home:** dataset summary and project overview.
- **Detection:** one-row manual input or batch CSV prediction. The single-row form requires all 30 features.
- **Analytics:** dataset and model metrics, ROC and precision-recall plots, and global feature importance.
- **Models:** detailed classifier metrics and confusion matrices for original and SMOTE training conditions.
- **About:** dataset, model, imbalance, and SHAP notes.

Accuracy can be misleading when fraud is rare. Review precision, recall, F1, PR-AUC, and the confusion matrix as well. The 0.50 cutoff is fixed in this application; predictions are estimates, not guarantees. SHAP explains the model's behavior, not the actual cause of a transaction.

## Main files and folders

```text
app.py                         Flask application and prediction routes
dataset/creditcard.csv         Source dataset (download separately; ignored by Git)
manual_test_data/              Two label-hidden held-out CSV examples
src/data_preprocessing.py      Dataset validation and reproducible split
src/train_models.py            Model fitting, comparison, and artifact generation
src/evaluate_models.py         Regenerates ROC and precision-recall plots
src/shap_explainer.py          Local and global SHAP explanations
templates/ and static/         Web pages, styles, and browser scripts
models/                        Fitted models and results (created during training)
artifacts/                     Metrics, plots, and SHAP outputs (created during training)
requirements.txt               Python dependencies
```
